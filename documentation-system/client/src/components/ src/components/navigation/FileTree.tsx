// src/components/navigation/FileTree.tsx
import React from 'react';
import TreeView from '@mui/lab/TreeView';
import TreeItem from '@mui/lab/TreeItem';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import { Documentation } from '../../types/documentation';

interface FileTreeProps {
  documentation: Documentation | null; // Allow for null documentation
  onFileSelect: (filePath: string | null) => void; // Allow null for deselection
}

interface TreeNode {
  id: string;
  name: string;
  file_path?: string;
  children?: TreeNode[];
}

export const FileTree: React.FC<FileTreeProps> = ({ documentation, onFileSelect }) => {
  const generateTreeData = (documentation: Documentation | null): TreeNode[] | undefined => {
    if (!documentation) {
      return undefined; // Or return an empty array if you prefer
    }

    const files = documentation.files || []; // Access the "files" property, defaulting to an empty array if it doesn't exist

    // Create a map to store nodes by their path
    const nodeMap: { [key: string]: TreeNode } = {};

    // Create root nodes for each top-level file/folder
    files.forEach(file => {
      const parts = file.file_path.split('/');
      let currentNode = nodeMap;
      parts.forEach((part, index) => {
        const currentPath = parts.slice(0, index + 1).join('/');
        if (!currentNode[currentPath]) {
          currentNode[currentPath] = {
            id: currentPath,
            name: part,
            file_path: index === parts.length - 1 ? file.file_path : undefined, // Only assign file_path to leaf nodes
            children: []
          };
        }
        currentNode = currentNode[currentPath].children as { [key: string]: TreeNode };
      });
    });

    // Convert the node map into an array of top-level nodes
    return Object.values(nodeMap) as TreeNode[];
  };

  const renderTree = (nodes: TreeNode) => (
    <TreeItem key={nodes.id} nodeId={nodes.id} label={nodes.name}>
      {Array.isArray(nodes.children) ? nodes.children.map((node) => renderTree(node)) : null}
    </TreeItem>
  );

  const treeData = generateTreeData(documentation);

  if (!treeData) {
    return <p>No file structure available.</p>;
  }

  return (
    <TreeView
      aria-label="file system navigator"
      defaultCollapseIcon={<ExpandMoreIcon />}
      defaultExpandIcon={<ChevronRightIcon />}
      sx={{ height: 240, flexGrow: 1, maxWidth: 400, overflowY: 'auto' }}
      onNodeSelect={(event, nodeId) => {
        const findNode = (nodes: TreeNode[], id: string): TreeNode | undefined => {
          for (const node of nodes) {
            if (node.id === id) {
              return node;
            }
            if (node.children) {
              const found = findNode(node.children, id);
              if (found) {
                return found;
              }
            }
          }
          return undefined;
        };

        const selectedNode = findNode(treeData, nodeId);
        onFileSelect(selectedNode?.file_path || null); // Pass null if no file_path
      }}
    >
      {treeData.map(node => renderTree(node))}
    </TreeView>
  );
};