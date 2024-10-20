// scripts/go_inserter.go

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"strings"
)

// InputData represents the structure of input JSON
type InputData struct {
	Code         string                 `json:"code"`
	Documentation map[string]interface{} `json:"documentation"`
	Language     string                 `json:"language"`
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
	Name      string    `json:"name"`
	Docstring string    `json:"docstring"`
	Methods   []Function `json:"methods"`
}

// Variable represents a variable in the code
type Variable struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Description string `json:"description"`
	File        string `json:"file"`
	Line        int    `json:"line"`
	Link        string `json:"link"`
	Example     string `json:"example"`
	References  string `json:"references"`
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
	Constants            []Variable  `json:"constants"`
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

	// Convert documentation map to Structure
	docBytes, err := json.Marshal(input.Documentation)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error marshaling documentation: %v\n", err)
		os.Exit(1)
	}
	if err := json.Unmarshal(docBytes, &structure); err != nil {
		fmt.Fprintf(os.Stderr, "Error unmarshaling documentation: %v\n", err)
		os.Exit(1)
	}

	// Traverse the AST to insert docstrings
	ast.Inspect(file, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.FuncDecl:
			// Insert function docstring
			for _, funcDoc := range structure.Functions {
				if node.Name.Name == funcDoc.Name {
					if funcDoc.Docstring != "" {
						if node.Doc == nil {
							node.Doc = &ast.CommentGroup{}
						}
						docComment := &ast.Comment{
							Text: fmt.Sprintf("// %s", funcDoc.Docstring),
						}
						node.Doc.List = append([]*ast.Comment{docComment}, node.Doc.List...)
					}
				}
			}
		case *ast.TypeSpec:
			// Insert type (struct) docstring
			for _, classDoc := range structure.Classes {
				if node.Name.Name == classDoc.Name {
					if node.Doc == nil {
						node.Doc = &ast.CommentGroup{}
					}
					docComment := &ast.Comment{
						Text: fmt.Sprintf("// %s", classDoc.Docstring),
					}
					node.Doc.List = append([]*ast.Comment{docComment}, node.Doc.List...)
					
					// Insert method docstrings
					if structType, ok := node.Type.(*ast.StructType); ok {
						// Methods are defined outside the struct; handle separately
					}
				}
			}
		}
		return true
	})

	// Note: Go does not have a direct equivalent to classes. Methods are associated with types (usually structs).
	// The above code handles inserting docstrings for functions and types.

	// Generate the modified code
	var modifiedCode strings.Builder
	if err := printer.Fprint(&modifiedCode, fset, file); err != nil {
		fmt.Fprintf(os.Stderr, "Error generating modified code: %v\n", err)
		os.Exit(1)
	}

	// Output the modified code
	fmt.Println(modifiedCode.String())
}
