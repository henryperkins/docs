// scripts/go_parser.go

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
)

// InputData represents the structure of input JSON
type InputData struct {
	Code     string `json:"code"`
	Language string `json:"language"`
}

// Function represents a function/method in the code
type Function struct {
	Name       string   `json:"name"`
	Docstring  string   `json:"docstring"`
	Args       []string `json:"args"`
	Async      bool     `json:"async"`
	Complexity int      `json:"complexity"` // Placeholder for cyclomatic complexity
}

// Class represents a class/type in the code
type Class struct {
	Name       string    `json:"name"`
	Docstring  string    `json:"docstring"`
	Methods    []Function `json:"methods"`
}

// Variable represents a variable in the code
type Variable struct {
	Name       string `json:"name"`
	Type       string `json:"type"`
	Description string `json:"description"`
	File       string `json:"file"`
	Line       int    `json:"line"`
	Link       string `json:"link"`
	Example    string `json:"example"`
	References string `json:"references"`
}

// Structure represents the overall code structure
type Structure struct {
	Summary              string      `json:"summary"`
	ChangesMade          []string    `json:"changes_made"`
	Functions            []Function  `json:"functions"`
	Classes              []Class     `json:"classes"`
	Halstead             map[string]float64 `json:"halstead"`
	MaintainabilityIndex float64     `json:"maintainability_index"`
	Variables            []Variable  `json:"variables"`
	Constants            []Variable  `json:"constants"` // Reusing Variable struct for constants
}

func main() {
	reader := bufio.NewReader(os.Stdin)
	inputBytes, err := reader.ReadBytes(0)
	if err != nil && err != os.EOF {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		os.Exit(1)
	}

	var input InputData
	if err := json.Unmarshal(inputBytes, &input); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing input JSON: %v\n", err)
		os.Exit(1)
	}

	if strings.ToLower(input.Language) != "go" {
		fmt.Fprintf(os.Stderr, "Unsupported language: %s\n", input.Language)
		os.Exit(1)
	}

	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "src.go", input.Code, parser.ParseComments)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing Go code: %v\n", err)
		os.Exit(1)
	}

	structure := Structure{
		Summary:              "", // To be filled externally or manually
		ChangesMade:          [], // To be filled externally or manually
		Functions:            []Function{},
		Classes:              []Class{},
		Halstead:             make(map[string]float64),
		MaintainabilityIndex: 0.0, // Placeholder
		Variables:            []Variable{},
		Constants:            []Variable{},
	}

	// Traverse the AST
	ast.Inspect(file, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.FuncDecl:
			funcInfo := Function{
				Name:      node.Name.Name,
				Docstring: getDocstring(node.Doc),
				Args:      getFuncArgs(node.Type.Params),
				Async:     false, // Go does not have async functions
				Complexity: 1,    // Placeholder for cyclomatic complexity
			}
			structure.Functions = append(structure.Functions, funcInfo)
		case *ast.TypeSpec:
			// Assuming structs as classes
			if structType, ok := node.Type.(*ast.StructType); ok {
				classInfo := Class{
					Name:      node.Name.Name,
					Docstring: getDocstring(node.Doc),
					Methods:   []Function{},
				}
				// Traverse methods
				ast.Inspect(file, func(n ast.Node) bool {
					if fn, ok := n.(*ast.FuncDecl); ok {
						if fn.Recv != nil && len(fn.Recv.List) > 0 {
							receiver := exprToString(fn.Recv.List[0].Type)
							if strings.Contains(receiver, classInfo.Name) {
								methodInfo := Function{
									Name:      fn.Name.Name,
									Docstring: getDocstring(fn.Doc),
									Args:      getFuncArgs(fn.Type.Params),
									Async:     false,
									Complexity: 1,
								}
								classInfo.Methods = append(classInfo.Methods, methodInfo)
							}
						}
					}
					return true
				})
				structure.Classes = append(structure.Classes, classInfo)
			}
		case *ast.ValueSpec:
			for i, name := range node.Names {
				varType := exprToString(node.Type)
				varDesc := ""
				if node.Comment != nil && len(node.Comment.List) > i {
					varDesc = strings.TrimPrefix(node.Comment.List[i].Text, "//")
				}
				variable := Variable{
					Name:        name.Name,
					Type:        varType,
					Description: varDesc,
					File:        "Unknown", // Can be set if file info is available
					Line:        fset.Position(name.Pos()).Line,
					Link:        "Unknown", // Can be constructed based on repository
					Example:     "No example provided.",
					References:  "No references.",
				}
				if strings.ToUpper(name.Name) == name.Name {
					structure.Constants = append(structure.Constants, variable)
				} else {
					structure.Variables = append(structure.Variables, variable)
				}
			}
		}
		return true
	})

	// Placeholder for Halstead metrics and Maintainability Index
	// These require detailed analysis and are not implemented here

	// Validate the structure against the schema
	validateStruct, err := json.Marshal(structure)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error marshaling structure: %v\n", err)
		os.Exit(1)
	}

	var schema ValidationSchema
	schema = loadSchema()

	var data interface{}
	if err := json.Unmarshal(validateStruct, &data); err != nil {
		fmt.Fprintf(os.Stderr, "Error unmarshaling structure: %v\n", err)
		os.Exit(1)
	}

	valid := validate(data)
	if !valid {
		fmt.Fprintf(os.Stderr, "Validation errors: %v\n", validate.errors)
		os.Exit(1)
	}

	// Output the structure as JSON
	fmt.Println(string(validateStruct))
}

// Helper functions

func getDocstring(doc *ast.CommentGroup) string {
	if doc == nil {
		return ""
	}
	return strings.TrimSpace(doc.Text())
}

func getFuncArgs(params *ast.FieldList) []string {
	if params == nil {
		return []string{}
	}
	args := []string{}
	for _, field := range params.List {
		for _, name := range field.Names {
			args = append(args, name.Name)
		}
	}
	return args
}

func exprToString(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + exprToString(t.X)
	case *ast.SelectorExpr:
		return exprToString(t.X) + "." + t.Sel.Name
	case *ast.ArrayType:
		return "[]" + exprToString(t.Elt)
	default:
		return "unknown"
	}
}

// Placeholder for schema loading and validation
// Implement schema loading and validation as needed
type ValidationSchema struct{}

func loadSchema() ValidationSchema {
	// Implement schema loading if necessary
	return ValidationSchema{}
}
