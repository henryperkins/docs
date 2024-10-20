// scripts/cpp_parser.cpp

#include <clang/AST/AST.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace clang;
using namespace clang::tooling;
using json = nlohmann::json;

// Structure Definitions
struct Function {
    std::string name;
    std::string docstring;
    std::vector<std::string> args;
    bool async;
    int complexity; // Placeholder
};

struct Class {
    std::string name;
    std::string docstring;
    std::vector<Function> methods;
};

struct Variable {
    std::string name;
    std::string type;
    std::string description;
    std::string file;
    int line;
    std::string link;
    std::string example;
    std::string references;
};

// Visitor Class
class CppASTVisitor : public RecursiveASTVisitor<CppASTVisitor> {
public:
    explicit CppASTVisitor(ASTContext *Context) : Context(Context) {}

    bool VisitCXXRecordDecl(CXXRecordDecl *Declaration) {
        if (Declaration->isThisDeclarationADefinition()) {
            Class cls;
            cls.name = Declaration->getNameAsString();
            cls.docstring = getDocstring(Declaration->getASTContext(), Declaration);

            for (auto method : Declaration->methods()) {
                Function func;
                func.name = method->getNameAsString();
                func.docstring = getDocstring(method->getASTContext(), method);
                func.async = false; // C++ does not have async functions
                func.complexity = 1; // Placeholder

                // Get function arguments
                for (auto param : method->parameters()) {
                    func.args.push_back(param->getNameAsString());
                }

                cls.methods.push_back(func);
            }

            classes.push_back(cls);
        }
        return true;
    }

    bool VisitVarDecl(VarDecl *Declaration) {
        if (Declaration->hasGlobalStorage()) {
            Variable var;
            var.name = Declaration->getNameAsString();
            var.type = Declaration->getType().getAsString();
            var.description = getDocstring(Declaration->getASTContext(), Declaration);
            var.file = Context->getSourceManager().getFilename(Declaration->getLocation()).str();
            var.line = Context->getSourceManager().getSpellingLineNumber(Declaration->getLocation());
            var.link = "Unknown"; // Construct based on repository URL
            var.example = "No example provided.";
            var.references = "No references.";

            // Determine if the variable is a constant (e.g., const or constexpr)
            QualType qt = Declaration->getType();
            if (qt.isConstQualified() || Declaration->isConstexpr()) {
                structure.constants.push_back(var);
            } else {
                structure.variables.push_back(var);
            }
        }
        return true;
    }

    json getJSON() {
        json j;
        j["summary"] = ""; // To be filled externally or manually
        j["changes_made"] = json::array(); // To be filled externally or manually
        j["functions"] = json::array(); // For standalone functions
        j["classes"] = json::array();
        j["halstead"] = {
            {"volume", 0.0},
            {"difficulty", 0.0},
            {"effort", 0.0}
        };
        j["maintainability_index"] = 0.0; // Placeholder
        j["variables"] = structure.variables;
        j["constants"] = structure.constants;

        for (const auto &cls : classes) {
            json jcls;
            jcls["name"] = cls.name;
            jcls["docstring"] = cls.docstring;
            jcls["methods"] = json::array();
            for (const auto &method : cls.methods) {
                json jmethod;
                jmethod["name"] = method.name;
                jmethod["docstring"] = method.docstring;
                jmethod["args"] = method.args;
                jmethod["async"] = method.async;
                jmethod["type"] = "instance"; // C++ does not have explicit method types
                jmethod["complexity"] = method.complexity;
                jcls["methods"].push_back(jmethod);
            }
            j["classes"].push_back(jcls);
        }

        // Add standalone functions
        // Placeholder: Implement extraction of standalone functions if necessary

        return j;
    }

private:
    ASTContext *Context;
    std::vector<Class> classes;

    struct InternalStructure {
        std::vector<Variable> variables;
        std::vector<Variable> constants;
    } structure;

    std::string getDocstring(ASTContext &Context, const Decl *Declaration) {
        std::string doc = "";
        RawComment *RC = Context.getRawCommentForAnyRedecl(Declaration);
        if (RC) {
            doc = RC->getRawText(Context.getSourceManager());
            // Clean up comment markers
            doc = std::regex_replace(doc, std::regex("^\\/\\/\\/\\s*"), "");
            doc = std::regex_replace(doc, std::regex("^\\/\\/\\s*"), "");
        }
        return doc;
    }
};

// AST Consumer
class CppASTConsumer : public ASTConsumer {
public:
    explicit CppASTConsumer(ASTContext *Context) : Visitor(Context) {}

    virtual void HandleTranslationUnit(ASTContext &Context) {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
        json j = Visitor.getJSON();
        std::cout << j.dump(4) << std::endl;
    }

private:
    CppASTVisitor Visitor;
};

// Frontend Action
class CppFrontendAction : public ASTFrontendAction {
public:
    virtual std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) {
        return std::make_unique<CppASTConsumer>(&CI.getASTContext());
    }
};

// Main Function
int main(int argc, const char **argv) {
    // Parse command-line options
    CommonOptionsParser OptionsParser(argc, argv, llvm::cl::GeneralCategory);
    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

    // Run the Clang Tool
    int result = Tool.run(newFrontendActionFactory<CppFrontendAction>().get());

    return result;
}
