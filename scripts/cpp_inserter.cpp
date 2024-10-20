// scripts/cpp_inserter.cpp

#include <clang/AST/AST.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <unordered_map>
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
    CppASTVisitor(ASTContext *Context, Rewriter &R, const std::unordered_map<std::string, std::string>& docMap)
        : Context(Context), TheRewriter(R), docMap(docMap) {}

    bool VisitCXXRecordDecl(CXXRecordDecl *Declaration) {
        if (Declaration->isThisDeclarationADefinition()) {
            std::string className = Declaration->getNameAsString();
            auto it = docMap.find(className);
            if (it != docMap.end()) {
                // Insert class docstring before the class declaration
                SourceLocation loc = Declaration->getBeginLoc();
                TheRewriter.InsertTextBefore(loc, "/**\n * " + it->second + "\n */\n");
            }

            for (auto method : Declaration->methods()) {
                std::string methodName = method->getNameAsString();
                std::string fullMethodName = className + "." + methodName;
                auto mit = docMap.find(fullMethodName);
                if (mit != docMap.end()) {
                    // Insert method docstring before the method declaration
                    SourceLocation loc = method->getBeginLoc();
                    TheRewriter.InsertTextBefore(loc, "/**\n * " + mit->second + "\n */\n");
                }
            }
        }
        return true;
    }

    bool VisitFunctionDecl(FunctionDecl *Declaration) {
        if (Declaration->isThisDeclarationADefinition() && !Declaration->isCXXClassMember()) {
            std::string funcName = Declaration->getNameAsString();
            auto it = docMap.find(funcName);
            if (it != docMap.end()) {
                // Insert function docstring before the function declaration
                SourceLocation loc = Declaration->getBeginLoc();
                TheRewriter.InsertTextBefore(loc, "/**\n * " + it->second + "\n */\n");
            }
        }
        return true;
    }

private:
    ASTContext *Context;
    Rewriter &TheRewriter;
    const std::unordered_map<std::string, std::string>& docMap;
};

// AST Consumer
class CppASTConsumer : public ASTConsumer {
public:
    CppASTConsumer(ASTContext *Context, Rewriter &R, const std::unordered_map<std::string, std::string>& docMap)
        : Visitor(Context, R, docMap) {}

    virtual void HandleTranslationUnit(ASTContext &Context) {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }

private:
    CppASTVisitor Visitor;
};

// Frontend Action
class CppFrontendAction : public ASTFrontendAction {
public:
    CppFrontendAction(const std::unordered_map<std::string, std::string>& docMap)
        : docMap(docMap) {}

    void EndSourceFileAction() override {
        SourceManager &SM = TheRewriter.getSourceMgr();
        llvm::outs() << TheRewriter.getEditBuffer(SM.getMainFileID()).Buf;
    }

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
        TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return std::make_unique<CppASTConsumer>(&CI.getASTContext(), TheRewriter, docMap);
    }

private:
    Rewriter TheRewriter;
    std::unordered_map<std::string, std::string> docMap;
};

// Main Function
int main(int argc, const char **argv) {
    // Read input from stdin
    std::string input;
    std::string line;
    while (std::getline(std::cin, line)) {
        input += line + "\n";
    }

    json inputData;
    try {
        inputData = json::parse(input);
    } catch (json::parse_error &e) {
        std::cerr << "Error parsing input JSON: " << e.what() << std::endl;
        return 1;
    }

    std::string code = inputData["code"].get<std::string>();
    std::unordered_map<std::string, std::string> documentation;

    // Parse documentation into a map
    if (inputData.contains("documentation")) {
        json doc = inputData["documentation"];
        // Classes
        if (doc.contains("classes")) {
            for (auto &cls : doc["classes"]) {
                std::string className = cls["name"].get<std::string>();
                std::string classDoc = cls["docstring"].get<std::string>();
                documentation[className] = classDoc;
                // Methods
                if (cls.contains("methods")) {
                    for (auto &method : cls["methods"]) {
                        std::string methodName = method["name"].get<std::string>();
                        std::string fullMethodName = className + "." + methodName;
                        std::string methodDoc = method["docstring"].get<std::string>();
                        documentation[fullMethodName] = methodDoc;
                    }
                }
            }
        }
        // Functions
        if (doc.contains("functions")) {
            for (auto &func : doc["functions"]) {
                std::string funcName = func["name"].get<std::string>();
                std::string funcDoc = func["docstring"].get<std::string>();
                documentation[funcName] = funcDoc;
            }
        }
    }

    // Create temporary file
    std::string tempFile = "temp.cpp";
    std::ofstream ofs(tempFile);
    ofs << code;
    ofs.close();

    // Parse command-line options
    CommonOptionsParser OptionsParser(argc, argv, llvm::cl::GeneralCategory);
    ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

    // Run the Clang Tool with our FrontendAction
    CppFrontendAction action(documentation);
    int result = Tool.run(newFrontendActionFactory(&action).get());

    // Cleanup temporary file
    remove(tempFile.c_str());

    return result;
}
