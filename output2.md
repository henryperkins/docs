# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/styles/globals.css

## Summary

This CSS file is responsible for importing the Notyf library's CSS styles, which are used for displaying notifications in a web application. The purpose of this file is to ensure that the notification styles are applied correctly throughout the application, enhancing the user experience with visually appealing alerts.

## Changes Made

- Added import statement for Notyf CSS styles.

```css
/* File: apps/frontend/src/styles/globals.css */

@import 'notyf/notyf.min.css';

/* ... existing styles ... */
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/styles/Header.css

## Summary

This CSS file defines the styling rules for the application header, ensuring a responsive and visually appealing layout. It utilizes flexbox for alignment and spacing, providing a modern design that enhances user experience.

## Changes Made

- Added styles for the application header using flexbox layout.
- Updated header text styles for improved readability.

```css
/* Styles the main header of the application, making it flexible and responsive. */
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background-color: #f8f9fa;
  border-bottom: 1px solid #e9ecef;
}

/* Styles the main title within the header, ensuring it has no margin and is appropriately sized for visibility. */
.app-header h1 {
  margin: 0;
  font-size: 1.5rem;
  color: #333;
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/styles/LogoutButton.css

## Summary

This CSS file defines styles for a logout button, ensuring it is visually appealing and user-friendly. The styles include padding, font size, background color, text color, border properties, and hover/focus effects to enhance user interaction.

## Changes Made

- Added styles for the logout button
- Updated hover and focus states for better accessibility

```css
/* Styles for the logout button, including padding, font size, background color, text color, border, border radius, cursor style, and transition effect. */
.btn-logout {
  padding: 0.5rem 1rem;
  font-size: 1rem;
  background-color: #f44336;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

/* Changes the background color of the logout button when hovered over to provide visual feedback. */
.btn-logout:hover {
  background-color: #d32f2f;
}

/* Removes the default outline and adds a box shadow when the logout button is focused, improving accessibility. */
.btn-logout:focus {
  outline: none;
  box-shadow: 0 0 0 2px rgba(244, 67, 54, 0.5);
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/components/SearchForm.css

## Summary

This CSS file is designed to style a search form and its results within a web application. It provides a clean and user-friendly interface for users to input search queries and view results, enhancing the overall user experience.

## Changes Made

- Added styles for the search form and its elements.
- Updated button styles for better visibility and interaction.
- Fixed input focus styles to improve accessibility.

```css
/* Search Form styles */
/* Styles the main search form container, providing padding, background color, and a bottom border. */
.search-form {
  padding: 15px 20px;
  background-color: #ffffff;
  border-bottom: 1px solid #e0e0e0;
}

/* Sets the form to use flexbox for layout, aligning items vertically in the center. */
.search-form form {
  display: flex;
  align-items: center;
}

/* Styles the input field to be flexible, with padding, font size, border, and rounded corners. */
.search-form input {
  flex: 1;
  padding: 10px 15px;
  font-size: 16px;
  border: 1px solid #e0e0e0;
  border-radius: 20px;
  outline: none;
}

/* Changes the border color of the input field when it is focused, enhancing visibility. */
.search-form input:focus {
  border-color: #007bff;
}

/* Styles the search button with no background or border, a larger font size, and a pointer cursor. */
.btn-search {
  margin-left: 10px;
  background: none;
  border: none;
  font-size: 20px;
  color: #333333;
  cursor: pointer;
}

/* Changes the text color of the search button on hover for better interactivity. */
.btn-search:hover {
  color: #007bff;
}

/* Styles the container for search results, providing padding for spacing. */
.search-results {
  padding: 20px;
}

/* Removes the top margin from the search results heading for better alignment. */
.search-results h3 {
  margin-top: 0;
}

/* Removes default list styles and padding/margin from the results list. */
.search-results ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

/* Adds spacing between individual search result items. */
.search-results li {
  margin-bottom: 10px;
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/styles/register.css

## Summary

This CSS file is designed to style the registration page of a web application. It provides a clean and modern layout for the registration form, ensuring that it is user-friendly and visually appealing. The styles include flexbox properties for centering the form, as well as specific styles for form elements, buttons, and text links.

## Changes Made

- Added styles for the registration page layout and form elements.
- Updated button styles to enhance user interaction.
- Fixed hover and disabled states for buttons.

```css
/* Styles the registration page to use flexbox for centering the content vertically and horizontally. */
.register-page {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #f5f5f5;
}

/* Defines the appearance of the registration form, including background color, padding, border radius, and shadow. */
.register-form {
  background-color: white;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 400px;
}

/* Centers the heading text within the registration form and adds bottom margin. */
.register-form h1 {
  text-align: center;
  margin-bottom: 1.5rem;
}

/* Adds bottom margin to each form group for spacing. */
.form-group {
  margin-bottom: 1rem;
}

/* Displays labels as block elements and adds margin below them. */
.form-group label {
  display: block;
  margin-bottom: 0.5rem;
}

/* Styles input fields to be full width with padding, font size, border, and border radius. */
.form-group input {
  width: 100%;
  padding: 0.5rem;
  font-size: 1rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}

/* Styles the registration button with full width, padding, background color, text color, and hover transition. */
.btn-register {
  width: 100%;
  padding: 0.75rem;
  font-size: 1rem;
  background-color: #0070f3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

/* Changes the background color of the button on hover for better user feedback. */
.btn-register:hover {
  background-color: #0051a2;
}

/* Styles the button when disabled, indicating it cannot be interacted with. */
.btn-register:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

/* Centers the redirect text and adds top margin. */
.redirect-text {
  text-align: center;
  margin-top: 1rem;
}

/* Styles links within the redirect text with color and removes underline. */
.redirect-text a {
  color: #0070f3;
  text-decoration: none;
}

/* Adds underline to links on hover for better visibility. */
.redirect-text a:hover {
  text-decoration: underline;
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/components/ConversationList.css

## Summary

This CSS file defines the styling for a conversation list component within a web application. It includes styles for the overall padding, headings, list items, and buttons, ensuring a clean and user-friendly interface for displaying conversations.

## Changes Made

- Added styles for padding in the conversation list.
- Updated margin for headings within the conversation list.
- Fixed list styles to remove default bullets and padding.
- Added margin to list items for better spacing.
- Defined button styles for a consistent look and feel.
- Added hover effect for buttons to enhance user interaction.

```css
/* Conversation List styles */
/* Sets the padding around the entire conversation list. */
.conversation-list {
  padding: 20px;
}

/* Removes the top margin for headings within the conversation list. */
.conversation-list h2 {
  margin-top: 0;
}

/* Removes default list styling, padding, and margin for unordered lists. */
.conversation-list ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

/* Adds bottom margin to list items for spacing. */
.conversation-list li {
  margin-bottom: 10px;
}

/* Styles buttons to be full-width, with no background or border, and adds padding and cursor styles. */
.conversation-list button {
  width: 100%;
  text-align: left;
  background: none;
  border: none;
  padding: 10px;
  font-size: 16px;
  cursor: pointer;
  border-radius: 5px;
  transition: background-color 0.2s;
}

/* Changes the background color of buttons on hover for visual feedback. */
.conversation-list button:hover {
  background-color: #f0f0f0;
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/styles/login.css

## Summary

This CSS file is designed to style a login page, providing a clean and user-friendly interface for users to log in. It utilizes flexbox for layout, ensuring that the login form is centered on the page, and includes styles for various elements such as form groups, buttons, and links.

## Changes Made

- Added styles for the login page layout.
- Updated button styles for hover and disabled states.
- Fixed alignment issues for form elements.

```css
/* Styles the login page to use flexbox for centering the content vertically and horizontally. */
.login-page {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #f5f5f5;
}

/* Styles the login form with a white background, padding, rounded corners, and a subtle shadow. */
.login-form {
  background-color: white;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 400px;
}

/* Centers the heading text within the login form and adds margin below it. */
.login-form h1 {
  text-align: center;
  margin-bottom: 1.5rem;
}

/* Adds margin below each form group for spacing. */
.form-group {
  margin-bottom: 1rem;
}

/* Displays labels as block elements with margin below for spacing. */
.form-group label {
  display: block;
  margin-bottom: 0.5rem;
}

/* Styles input fields to be full width with padding, border, and rounded corners. */
.form-group input {
  width: 100%;
  padding: 0.5rem;
  font-size: 1rem;
  border: 1px solid #ccc;
  border-radius: 4px;
}

/* Styles the login button with full width, padding, background color, and hover transition. */
.btn-login {
  width: 100%;
  padding: 0.75rem;
  font-size: 1rem;
  background-color: #0070f3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

/* Changes the background color of the login button on hover. */
.btn-login:hover {
  background-color: #0051a2;
}

/* Styles the disabled state of the login button with a grey background and not-allowed cursor. */
.btn-login:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

/* Centers the redirect text and adds margin above it. */
.redirect-text {
  text-align: center;
  margin-top: 1rem;
}

/* Styles links within the redirect text with a specific color and removes underline. */
.redirect-text a {
  color: #0070f3;
  text-decoration: none;
}

/* Adds underline to links on hover. */
.redirect-text a:hover {
  text-decoration: underline;
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/components/FileUploadForm.css

## Summary

This CSS file defines the styles for a file upload form and its associated elements, ensuring a user-friendly interface with a clean and modern design. It includes styles for form layout, input fields, buttons, and result display, contributing to the overall aesthetic and usability of the application.

## Changes Made

- Added styles for the file upload form
- Updated button hover effects
- Fixed input field focus styles

```css
/* File: apps/frontend/src/components/FileUploadForm.css */

/* Styles the main container of the file upload form, providing a clean background, padding, rounded corners, and a subtle shadow. */
.file-upload-form {
  background-color: #ffffff;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  margin-bottom: 20px;
}

/* Styles the heading of the file upload form, setting a bottom margin and a dark text color. */
.file-upload-form h2 {
  margin-bottom: 15px;
  color: #333333;
}

/* Adds spacing between form groups to improve layout. */
.file-upload-form .form-group {
  margin-bottom: 15px;
}

/* Styles the labels within the form, making them bold and ensuring they are displayed as block elements for proper alignment. */
.file-upload-form label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
  color: #555555;
}

/* Styles the file input field to be full-width with padding, a border, and rounded corners. */
.file-upload-form input[type="file"] {
  width: 100%;
  padding: 8px;
  border: 1px solid #cccccc;
  border-radius: 4px;
  outline: none;
}

/* Changes the border color of the file input field when it is focused, enhancing user interaction. */
.file-upload-form input[type="file"]:focus {
  border-color: #007bff;
}

/* Styles the upload button with a green background, white text, and rounded corners, making it prominent and clickable. */
.file-upload-form .btn-upload {
  width: 100%;
  padding: 10px;
  background-color: #28a745; /* Green button color */
  color: #ffffff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

/* Changes the background color of the upload button on hover for better visual feedback. */
.file-upload-form .btn-upload:hover {
  background-color: #218838;
}

/* Adds top margin to the analysis result section to separate it from the upload form. */
.analysis-result {
  margin-top: 20px;
}

/* Styles the heading of the analysis result section with a bottom margin and dark text color. */
.analysis-result h3 {
  margin-bottom: 10px;
  color: #333333;
}

/* Styles the paragraph within the analysis result section with a light background, padding, rounded corners, and a gray text color. */
.analysis-result p {
  background-color: #f8f9fa;
  padding: 10px;
  border-radius: 4px;
  color: #555555;
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/components/FewShotForm.css

## Summary

This CSS file defines the styles for a few-shot form component, ensuring a clean and user-friendly interface. It includes styles for layout, typography, and interactive elements, enhancing the overall user experience within the application.

## Changes Made

- Added styles for the few-shot form component
- Updated input focus styles
- Fixed button hover states

```css
/* Few-Shot Form styles */
/* Styles the main container of the few-shot form, providing padding, background color, border, and margin. */
.few-shot-form {
  padding: 20px;
  background-color: #ffffff;
  border: 1px solid #e0e0e0;
  margin: 20px 0;
}

/* Removes the top margin from the h2 element within the few-shot form. */
.few-shot-form h2 {
  margin-top: 0;
}

/* Adds bottom margin to each form group for spacing. */
.few-shot-form .form-group {
  margin-bottom: 15px;
}

/* Styles labels to be block elements with bottom margin and bold font weight. */
.few-shot-form label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

/* Styles input fields to be full-width with padding, font size, border, and rounded corners. */
.few-shot-form input {
  width: 100%;
  padding: 10px;
  font-size: 16px;
  border: 1px solid #e0e0e0;
  border-radius: 5px;
  outline: none;
}

/* Changes the border color of input fields when focused. */
.few-shot-form input:focus {
  border-color: #007bff;
}

/* Styles the add button with padding, font size, background color, text color, border, and cursor. */
.few-shot-form .btn-add {
  padding: 10px 20px;
  font-size: 16px;
  background-color: #007bff;
  color: #ffffff;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

/* Changes the background color of the add button on hover. */
.few-shot-form .btn-add:hover {
  background-color: #0056b3;
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/next-env.d.ts

## Summary

This JavaScript file serves as a foundational structure for a larger application, currently containing no functions or classes. It is designed to be extended with additional functionality as the application evolves.

## Changes Made



```typescript
/// <reference types="next" />
/// <reference types="next/image-types/global" />

// NOTE: This file should not be edited
// see https://nextjs.org/docs/basic-features/typescript for more information.
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/styles/global.css

## Summary

This CSS file provides styling for a web application, focusing on layout, typography, and responsive design. It defines styles for various components such as the home page, search forms, results, and file upload forms, ensuring a consistent and user-friendly interface.

## Changes Made

- Added base styles for body and universal selectors.
- Updated styles for search forms and results to enhance usability.
- Fixed responsive design issues for side panels on smaller screens.

```css
/* File: apps/frontend/src/styles/globals.css */

/* Reset styles */
/* Resets default margin and padding for all elements and sets box-sizing to border-box. */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

/* Global styles */
/* Sets the default font family, background color, and text color for the body. */
body {
  font-family: 'Helvetica Neue', Arial, sans-serif;
  background-color: #f5f5f5;
  color: #333333;
}

/* Home Page Layout */
/* Defines the layout for the home page as a flex container with a column direction. */
.home-page {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

/* Sets the main content area to be a flex container that takes up available space. */
.main-content {
  display: flex;
  flex: 1;
}

/* Styles the side panel with a fixed width, background color, border, and padding. */
.side-panel {
  width: 300px;
  background-color: #ffffff;
  border-left: 1px solid #e0e0e0;
  overflow-y: auto;
  padding: 20px;
}

/* Search Form Styles */
/* Styles the search form with padding, background color, and a bottom border. */
.search-form {
  padding: 15px 20px;
  background-color: #ffffff;
  border-bottom: 1px solid #e0e0e0;
}

/* Sets the search form to use flexbox for layout and centers items vertically. */
.search-form form {
  display: flex;
  align-items: center;
}

/* Styles the input field within the search form for better usability. */
.search-form input {
  flex: 1;
  padding: 10px 15px;
  font-size: 16px;
  border: 1px solid #e0e0e0;
  border-radius: 20px;
  outline: none;
}

/* Changes the border color of the input field when focused. */
.search-form input:focus {
  border-color: #007bff;
}

/* Styles the search button with no background or border, and sets its font size. */
.btn-search {
  margin-left: 10px;
  background: none;
  border: none;
  font-size: 20px;
  color: #333333;
  cursor: pointer;
}

/* Changes the text color of the search button on hover. */
.btn-search:hover {
  color: #007bff;
}

/* Styles the loading indicator for search results. */
.search-loading {
  padding: 10px 20px;
  color: #555555;
}

/* Styles the container for search results. */
.search-results {
  padding: 20px;
}

/* Styles the heading within search results. */
.search-results h3 {
  margin-bottom: 10px;
  color: #333333;
}

/* Removes default list styling for search results. */
.search-results ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

/* Adds margin to each list item in search results. */
.search-results li {
  margin-bottom: 10px;
}

/* Styles buttons in search results for better interaction. */
.search-results button {
  width: 100%;
  text-align: left;
  background: none;
  border: none;
  padding: 10px;
  font-size: 16px;
  cursor: pointer;
  border-radius: 5px;
  transition: background-color 0.2s;
}

/* Changes background color of buttons on hover. */
.search-results button:hover {
  background-color: #f0f0f0;
}

/* Few-Shot Form Styles */
/* Styles the few-shot form with padding, border, and background. */
.few-shot-form {
  padding: 20px;
  background-color: #ffffff;
  border: 1px solid #e0e0e0;
  margin-bottom: 20px;
  border-radius: 5px;
}

/* Styles the heading within the few-shot form. */
.few-shot-form h2 {
  margin-top: 0;
  color: #333333;
}

/* Adds margin to form groups within the few-shot form. */
.few-shot-form .form-group {
  margin-bottom: 15px;
}

/* Styles labels within the few-shot form. */
.few-shot-form label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
  color: #555555;
}

/* Styles input fields within the few-shot form. */
.few-shot-form input {
  width: 100%;
  padding: 10px 15px;
  font-size: 16px;
  border: 1px solid #e0e0e0;
  border-radius: 5px;
  outline: none;
}

/* Changes border color of input fields when focused. */
.few-shot-form input:focus {
  border-color: #007bff;
}

/* Styles the add button with padding and background color. */
.btn-add {
  padding: 10px 20px;
  font-size: 16px;
  background-color: #28a745; /* Green button color */
  color: #ffffff;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

/* Changes background color of the add button on hover. */
.btn-add:hover {
  background-color: #218838;
}

/* File Upload Form Styles */
/* Styles the file upload form. */
.file-upload-form {
  padding: 20px;
  background-color: #ffffff;
  border: 1px solid #e0e0e0;
  border-radius: 5px;
}

/* Styles the heading within the file upload form. */
.file-upload-form h2 {
  margin-top: 0;
  color: #333333;
}

/* Adds margin to form groups within the file upload form. */
.file-upload-form .form-group {
  margin-bottom: 15px;
}

/* Styles labels within the file upload form. */
.file-upload-form label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
  color: #555555;
}

.file-upload-form input[type='file'] {
  font-size: 16px;
}

/* Styles the upload button. */
.btn-upload {
  padding: 10px 20px;
  font-size: 16px;
  background-color: #17a2b8; /* Teal button color */
  color: #ffffff;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

/* Changes background color of the upload button on hover. */
.btn-upload:hover {
  background-color: #138496;
}

/* Styles the analysis result section. */
.analysis-result {
  margin-top: 20px;
  background-color: #e9ecef;
  padding: 15px;
  border-radius: 5px;
}

/* Styles the heading within the analysis result. */
.analysis-result h3 {
  margin-bottom: 10px;
  color: #333333;
}

/* Styles paragraphs within the analysis result. */
.analysis-result p {
  color: #555555;
}
/* File: apps/frontend/src/styles/globals.css */

/* Home Page Layout */
/* Defines the layout for the home page as a flex container with a column direction. */
.home-page {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

/* Styles the header of the home page. */
.home-header {
  padding: 15px 20px;
  background-color: #ffffff;
  border-bottom: 1px solid #e0e0e0;
}

/* Sets the main content area to be a flex container that takes up available space. */
.main-content {
  display: flex;
  flex: 1;
}

/* Sets the chat main area to take up available space. */
.main-content .chat-main {
  flex: 1;
}

/* Styles the side panel with a fixed width, background color, border, and padding. */
.side-panel {
  width: 300px;
  background-color: #ffffff;
  border-left: 1px solid #e0e0e0;
  overflow-y: auto;
  padding: 20px;
}

/* Media query to hide the side panel on screens smaller than 768px. */
@media (max-width: 768px) {
/* Styles the side panel with a fixed width, background color, border, and padding. */
  .side-panel {
    display: none; /* Hide side panel on small screens */
  }
}

```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/pages/login.tsx

## Summary

This JavaScript code defines a LoginPage component and a handleLogin function, which are part of a user authentication system in a React application. The LoginPage component is responsible for rendering the login interface, while the handleLogin function processes the login form submission.

## Changes Made

- Added LoginPage component for user authentication.
- Implemented handleLogin function to manage login events.

```typescript
import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';
import { API_BASE_URL } from '../utils/config';
import './login.css'; // Create this CSS file for styling

const notyf = new Notyf();

const LoginPage: React.FC = () => {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState<boolean>(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!email.trim() || !password.trim()) {
      notyf.error('Email and password are required.');
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email: email.trim(), password: password.trim() }),
      });

      const data = await response.json();

      if (response.ok) {
        notyf.success('Logged in successfully.');
        router.push('/chat'); // Redirect to chat page
      } else {
        notyf.error(data.message || 'Login failed.');
      }
    } catch (error: any) {
      notyf.error(error.message || 'Login failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <form className="login-form" onSubmit={handleLogin}>
        <h1>Login</h1>
        <div className="form-group">
          <label htmlFor="email">Email:</label>
          <input
            type="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email"
            aria-label="Email"
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="password">Password:</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter your password"
            aria-label="Password"
            required
          />
        </div>
        <button type="submit" className="btn btn-login" disabled={loading}>
          {loading ? 'Logging in...' : 'Login'}
        </button>
        <p className="redirect-text">
          Don't have an account? <a href="/register">Register here</a>
        </p>
      </form>
    </div>
  );
};

export default LoginPage;
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/components/LoginForm.tsx

## Summary

This JavaScript code defines a login form component and its associated login handling functionality, primarily designed for a React application. It manages user input and processes login events, ensuring a smooth user experience during authentication.

## Changes Made

- Added LoginForm component to handle user login interface.
- Implemented handleLogin function to process login events and manage form submissions.

```typescript
// File: apps/frontend/src/components/LoginForm.tsx

import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { saveToken } from '../utils/auth';
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';

const notyf = new Notyf();

const LoginForm: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState<boolean>(false);
  const router = useRouter();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!email.trim() || !password.trim()) {
      notyf.error('Please enter both email and password.');
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/api/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email: email.trim(), password: password.trim() }),
      });

      const data = await response.json();

      if (response.ok) {
        saveToken(data.token);
        notyf.success('Logged in successfully.');
        router.push('/');
      } else {
        notyf.error(data.message || 'Login failed.');
      }
    } catch (error: any) {
      notyf.error(error.message || 'Login failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="login-form" onSubmit={handleLogin}>
      <h1>Login</h1>
      <div className="form-group">
        <label htmlFor="email">Email:</label>
        <input
          type="email"
          id="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Enter your email"
          aria-label="Email"
          required
        />
      </div>
      <div className="form-group">
        <label htmlFor="password">Password:</label>
        <input
          type="password"
          id="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Enter your password"
          aria-label="Password"
          required
        />
      </div>
      <button type="submit" className="btn btn-login" disabled={loading}>
        {loading ? 'Logging in...' : 'Login'}
      </button>
    </form>
  );
};

export default LoginForm;
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/types/index.ts

## Summary

This JavaScript file serves as a foundational module for the application, providing essential functionalities and structures that can be utilized across various components. Currently, it does not contain any defined functions or classes, indicating that it may be a placeholder for future development or an empty template for further enhancements.

## Changes Made



```typescript
export interface Message {
  id: string;
  sender: 'user' | 'bot';
  content: string;
  timestamp: number;
}

export interface AnalysisResult {
  analysis: string;
}

export interface ErrorResponse {
  error: string;
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/pages/register.css

## Summary

This CSS file is designed to style a registration page, providing a clean and user-friendly interface for users to create an account. It includes styles for the overall layout, form elements, buttons, and responsive design for various screen sizes.

## Changes Made

- Added styles for the registration page layout
- Updated button styles for hover and disabled states
- Fixed responsive styles for smaller screens

```css
/* File: apps/frontend/src/pages/register.css */

/* Styles the main registration page container to center its content both vertically and horizontally. */
.register-page {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background-color: #f5f5f5;
  padding: 20px;
}

/* Styles the registration form with a white background, padding, rounded corners, and a subtle shadow. */
.register-form {
  background-color: #ffffff;
  padding: 30px;
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 400px;
}

/* Styles the heading of the registration form to be centered with a specific color and margin. */
.register-form h1 {
  margin-bottom: 20px;
  text-align: center;
  color: #333333;
}

/* Adds margin to the bottom of each form group for spacing. */
.register-form .form-group {
  margin-bottom: 15px;
}

/* Styles the labels within the form to be block elements with bold text. */
.register-form label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
  color: #555555;
}

/* Styles the input fields to be full width with padding, border, and rounded corners. */
.register-form input {
  width: 100%;
  padding: 10px 15px;
  font-size: 16px;
  border: 1px solid #cccccc;
  border-radius: 5px;
  outline: none;
}

/* Changes the border color of the input fields when focused. */
.register-form input:focus {
  border-color: #007bff;
}

/* Styles the registration button with a green background, white text, and rounded corners. */
.btn-register {
  width: 100%;
  padding: 10px 0;
  font-size: 16px;
  background-color: #28a745; /* Green button color */
  color: #ffffff;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

/* Changes the background color of the button on hover. */
.btn-register:hover {
  background-color: #218838;
}

/* Styles the button when disabled to indicate it cannot be clicked. */
.btn-register:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

/* Styles the redirect text to be centered with a specific color. */
.redirect-text {
  margin-top: 15px;
  text-align: center;
  color: #555555;
}

/* Styles the links within the redirect text. */
.redirect-text a {
  color: #007bff;
  text-decoration: none;
}

/* Underlines the link text on hover. */
.redirect-text a:hover {
  text-decoration: underline;
}

/* Responsive design */
/* Adjusts styles for screens smaller than 480px, reducing padding and font sizes. */
@media (max-width: 480px) {
/* Styles the registration form with a white background, padding, rounded corners, and a subtle shadow. */
  .register-form {
    padding: 20px;
  }

/* Styles the input fields to be full width with padding, border, and rounded corners. */
  .register-form input {
    font-size: 14px;
  }

/* Styles the registration button with a green background, white text, and rounded corners. */
  .btn-register {
    font-size: 14px;
  }
}

/* Further adjusts styles for screens smaller than 320px, reducing font sizes for better readability. */
@media (max-width: 320px) {
/* Styles the heading of the registration form to be centered with a specific color and margin. */
  .register-form h1 {
    font-size: 24px;
  }

/* Styles the input fields to be full width with padding, border, and rounded corners. */
  .register-form input {
    font-size: 12px;
  }

/* Styles the registration button with a green background, white text, and rounded corners. */
  .btn-register {
    font-size: 12px;
  }
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/utils/pusher.ts

## Summary

This JavaScript file contains functionality for initializing a Pusher instance, which is used for real-time communication in web applications. The code is designed to facilitate the connection to a Pusher service, allowing for the handling of real-time events and updates within the application.

## Changes Made

- Added function to initialize Pusher instance.

```typescript
// File: apps/frontend/src/utils/pusher.ts

import Pusher from 'pusher-js';

export const initializePusher = () => {
  const pusher = new Pusher(process.env.NEXT_PUBLIC_PUSHER_KEY || '', {
    cluster: process.env.NEXT_PUBLIC_PUSHER_CLUSTER || '',
  });
  return pusher;
};
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/pages/login.css

## Summary

This CSS file is designed to style the login page of a web application. It provides a clean and modern layout for the login form, ensuring that it is user-friendly and visually appealing across different devices.

## Changes Made

- Added styles for the login page layout and form elements.
- Updated button styles for better user interaction.
- Fixed responsive design issues for smaller screens.

```css
/* File: apps/frontend/src/pages/login.css */

/* Styles the main login page to center the content vertically and horizontally, with a light gray background. */
.login-page {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
  background-color: #f5f5f5;
  padding: 20px;
}

/* Defines the appearance of the login form, including background color, padding, border radius, and shadow for depth. */
.login-form {
  background-color: #ffffff;
  padding: 30px;
  border-radius: 10px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: 400px;
}

/* Styles the heading of the login form to be centered with a specific color and margin. */
.login-form h1 {
  margin-bottom: 20px;
  text-align: center;
  color: #333333;
}

/* Adds spacing between form groups for better layout. */
.login-form .form-group {
  margin-bottom: 15px;
}

/* Styles the labels within the form to be bold and spaced appropriately. */
.login-form label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
  color: #555555;
}

/* Styles the input fields to be full-width with padding, border, and rounded corners. */
.login-form input {
  width: 100%;
  padding: 10px 15px;
  font-size: 16px;
  border: 1px solid #cccccc;
  border-radius: 5px;
  outline: none;
}

/* Changes the border color of input fields when focused to indicate active selection. */
.login-form input:focus {
  border-color: #007bff;
}

/* Styles the login button with a blue background, white text, and rounded corners. */
.btn-login {
  width: 100%;
  padding: 10px 0;
  font-size: 16px;
  background-color: #007bff;
  color: #ffffff;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

/* Changes the background color of the button on hover for better interactivity. */
.btn-login:hover {
  background-color: #0069d9;
}

/* Styles the button when disabled to indicate it cannot be interacted with. */
.btn-login:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

/* Styles the redirect text below the login form for better visibility. */
.redirect-text {
  margin-top: 15px;
  text-align: center;
  color: #555555;
}

/* Styles the links within the redirect text to be green and without underline. */
.redirect-text a {
  color: #28a745;
  text-decoration: none;
}

/* Adds underline to links on hover for better user experience. */
.redirect-text a:hover {
  text-decoration: underline;
}

/* Responsive design */
/* Adjusts styles for devices with a maximum width of 480px to improve usability. */
@media (max-width: 480px) {
/* Defines the appearance of the login form, including background color, padding, border radius, and shadow for depth. */
  .login-form {
    padding: 20px;
  }

/* Styles the input fields to be full-width with padding, border, and rounded corners. */
  .login-form input {
    font-size: 14px;
  }

/* Styles the login button with a blue background, white text, and rounded corners. */
  .btn-login {
    font-size: 14px;
  }
}

/* Further adjusts styles for devices with a maximum width of 320px for optimal viewing. */
@media (max-width: 320px) {
/* Styles the heading of the login form to be centered with a specific color and margin. */
  .login-form h1 {
    font-size: 24px;
  }

/* Styles the input fields to be full-width with padding, border, and rounded corners. */
  .login-form input {
    font-size: 12px;
  }

/* Styles the login button with a blue background, white text, and rounded corners. */
  .btn-login {
    font-size: 12px;
  }
}

```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/components/FewShotForm.tsx

## Summary

This JavaScript file contains a React component named `FewShotForm` that handles the submission of a form. It is designed to facilitate few-shot learning tasks by allowing users to input data and submit it for processing. The component manages form state and submission events, ensuring a smooth user experience within the larger application context that likely involves machine learning or data processing.

## Changes Made

- Added FewShotForm component to handle form submissions.
- Implemented handleSubmit function to manage form submission logic.

```typescript
import React, { useState } from 'react';
import { Notyf } from 'notyf';

const notyf = new Notyf();

const FewShotForm: React.FC = () => {
  const [examples, setExamples] = useState<string>('');

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!examples.trim()) {
      notyf.error('Please provide at least one example');
      return;
    }

    try {
      // TODO: Implement the API call to send few-shot examples
      console.log('Few-shot examples:', examples);
      notyf.success('Few-shot examples submitted successfully');
      setExamples('');
    } catch (error) {
      notyf.error('An error occurred while submitting few-shot examples');
    }
  };

  return (
    <div className="few-shot-form">
      <h2>Few-Shot Learning</h2>
      <form onSubmit={handleSubmit}>
        <textarea
          value={examples}
          onChange={(e) => setExamples(e.target.value)}
          placeholder="Enter few-shot examples, one per line"
          rows={5}
        />
        <button type="submit">Submit Examples</button>
      </form>
    </div>
  );
};

export default FewShotForm;
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/components/Header.tsx

## Summary

This JavaScript file contains utility functions for managing headers and authentication checks within a web application. The functions are designed to streamline the process of handling user authentication and setting HTTP headers for requests, ensuring secure and efficient communication with the server.

## Changes Made

- Added Header function for managing HTTP headers
- Added checkAuth function to verify user authentication

```typescript
import React, { useContext, useEffect, useState } from 'react';
import LogoutButton from './LogoutButton';
import { ConversationContext } from '../contexts/ConversationContext';
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';
import './Header.css'; // Create this CSS file for styling

const notyf = new Notyf();

const Header: React.FC = () => {
  const { conversationId } = useContext(ConversationContext);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);

  useEffect(() => {
    // Simple check for authentication status
    // This can be enhanced by verifying token validity
    const checkAuth = async () => {
      try {
        const response = await fetch('/api/protected_route', { method: 'GET' });
        if (response.ok) {
          setIsAuthenticated(true);
        } else {
          setIsAuthenticated(false);
        }
      } catch {
        setIsAuthenticated(false);
      }
    };

    checkAuth();
  }, []);

  return (
    <header className="app-header">
      <h1>ChatApp Vercel-D</h1>
      {isAuthenticated && <LogoutButton />}
    </header>
  );
};

export default Header;
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/config.ts

## Summary

This JavaScript file serves as a foundational structure for a larger application, currently containing no defined functions or classes. It is designed to be expanded upon as development progresses, allowing for the integration of various functionalities and features as needed.

## Changes Made



```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:3000';

export default {
  API_BASE_URL,
};
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/jest.config.js

## Summary

This JavaScript file serves as a foundational module for a larger application, providing essential functionalities and structures that can be utilized by other components. It is designed to be modular and reusable, ensuring that the application remains maintainable and scalable.

## Changes Made



```javascript
// File: apps/backend/jest.config.js

module.exports = {
    preset: 'ts-jest',
    testEnvironment: 'node',
    testPathIgnorePatterns: ['/node_modules/', '/.next/'],
  };
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/components/SearchForm.tsx

## Summary

This JavaScript code defines a search form component that handles user input and submission events within a React application. It is designed to facilitate searching functionality, allowing users to submit queries and receive results based on their input.

## Changes Made

- Added SearchForm component to manage search functionality.
- Implemented handleSubmit function to handle form submission events.

```typescript
import React, { useState } from 'react';
import fetchWithAuth from '../utils/fetchWithAuth';
import { Notyf } from 'notyf';

const notyf = new Notyf();

interface SearchResult {
  id: string;
  content: string;
}

const SearchForm: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!query.trim()) {
      notyf.error('Please enter a search query');
      return;
    }

    try {
      const response = await fetchWithAuth(`/api/search?q=${encodeURIComponent(query)}`, {
        method: 'GET',
      });

      if (response.ok) {
        const data = await response.json();
        setResults(data.results);
      } else {
        notyf.error('Failed to perform search');
      }
    } catch (error) {
      notyf.error('An error occurred while searching');
    }
  };

  return (
    <div className="search-form">
      <h2>Search Conversations</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter search query"
        />
        <button type="submit">Search</button>
      </form>
      {results.length > 0 && (
        <div className="search-results">
          <h3>Search Results:</h3>
          <ul>
            {results.map((result) => (
              <li key={result.id}>{result.content}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default SearchForm;
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/pages/api/logout.ts

## Summary

This JavaScript file contains an API handler function designed to process incoming requests and send appropriate responses. It is part of a larger application that utilizes Next.js for server-side rendering and API routes.

## Changes Made

- Added the handler function to process API requests.

```typescript
import { NextApiRequest, NextApiResponse } from 'next';
import { removeTokenCookie } from '@/utils/auth';
import { apiHandler } from '@/utils/apiHandler';

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  if (req.method !== 'POST') {
    throw { statusCode: 405, message: `Method ${req.method} Not Allowed` };
  }

  removeTokenCookie(res);
  res.status(200).json({ message: 'Logged out successfully.' });
};

export default apiHandler(handler);
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/utils/azure.ts

## Summary

This JavaScript code provides utility functions for managing chat interactions with Azure's API. It includes functionalities for sending messages, counting tokens, managing context windows, and analyzing file content. These functions are essential for ensuring efficient communication and data handling in chat applications that utilize Azure's services.

## Changes Made

- Added getAzureResponse function to handle API responses from Azure.
- Added countTokens function to count the number of tokens in a given text.
- Added manageContextWindow function to manage the context window for chat messages.
- Added analyzeFileContent function to analyze the content of a specified file.

```typescript
import { Configuration, OpenAIApi } from 'openai';
import { OpenAIClient, AzureKeyCredential } from '@azure/openai';
import fs from 'fs';
import path from 'path';

// Configure the OpenAI API client
const configuration = new Configuration({
  apiKey: process.env.AZURE_API_KEY,
  basePath: process.env.AZURE_API_URL,
});
const openai = new OpenAIApi(configuration);

// Configure the Azure OpenAI client
const endpoint = process.env.AZURE_OPENAI_ENDPOINT!;
const apiKey = process.env.AZURE_OPENAI_API_KEY!;
const client = new OpenAIClient(endpoint, new AzureKeyCredential(apiKey));

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

// Function to get a chat completion from Azure OpenAI
export const getAzureResponse = async (
  messages: ChatMessage[],
  max_tokens: number = 1000,
  temperature: number = 0.7,
  top_p: number = 1
): Promise<string> => {
  try {
    const response = await openai.createChatCompletion({
      model: process.env.AZURE_DEPLOYMENT_NAME || '',
      messages,
      max_tokens,
      temperature,
      top_p,
    });

    return response.data.choices[0].message?.content || ''; 
  } catch (error) {
    console.error('Error getting response from Azure:', error);
    throw error;
  }
};

// Function to count tokens (using a simple approximation for now)
export const countTokens = (text: string): number => {
  return text.split(/\s+/).length; 
};

// Function to manage the context window (basic implementation)
export const manageContextWindow = (messages: ChatMessage[], maxTokens: number): ChatMessage[] => {
  let totalTokens = 0;
  const contextWindow: ChatMessage[] = [];

  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    const messageTokens = countTokens(message.content);

    if (totalTokens + messageTokens <= maxTokens) {
      contextWindow.unshift(message);
      totalTokens += messageTokens;
    } else {
      break;
    }
  }

  return contextWindow;
};

// Function to analyze file content
export async function analyzeFileContent(filePath: string): Promise<string> {
  try {
    const absolutePath = path.resolve(filePath);
    const fileContent = fs.readFileSync(absolutePath, 'utf-8');

    const prompt = `Analyze the following content:\n\n${fileContent}`;

    const deploymentName = process.env.AZURE_DEPLOYMENT_NAME || '';
    const result = await client.getCompletions(deploymentName, prompt, {
      maxTokens: 150,
      temperature: 0.7,
    });

    const analysis = result.choices[0].text.trim();
    return analysis;
  } catch (error) {
    console.error('Error in analyzeFileContent:', error);
    throw new Error('Failed to analyze file content');
  }
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/utils/apiHandler.ts

## Summary

This JavaScript file contains a function named `apiHandler` which is designed to handle API requests by accepting a handler function as a parameter. It serves as a middleware or utility function within a larger application context, facilitating the processing of API calls.

## Changes Made

- Added the `apiHandler` function to handle API requests.

```typescript
import { NextApiRequest, NextApiResponse } from 'next';

export function apiHandler(handler: Function) {
  return async (req: NextApiRequest, res: NextApiResponse) => {
    try {
      await handler(req, res);
    } catch (error) {
      console.error('API Error:', error);
      const statusCode = error.statusCode || 500;
      const message = error.message || 'An unexpected error occurred';
      res.status(statusCode).json({ error: message });
    }
  };
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/components/Chat.tsx

## Summary

This JavaScript code is designed to manage a chat application, providing functionalities such as sending messages, toggling the sidebar, and handling user input. It serves as the core logic for user interactions within the chat interface, ensuring a smooth and responsive experience.

## Changes Made

- Added function to handle user input events.
- Updated message sending logic to include error handling.
- Fixed sidebar toggle functionality to improve user experience.

```typescript
import React, { useState, useEffect, useRef, useContext } from 'react';
import Pusher from 'pusher-js';
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';
import './Chat.css';
import fetchWithAuth from '../utils/fetchWithAuth';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBars, faPlus, faRedo, faPaperPlane, faHistory, faUpload, faLightbulb, faSearch } from '@fortawesome/free-solid-svg-icons';
import ConversationList from './ConversationList';
import { ConversationContext } from '../contexts/ConversationContext';
import { countTokens } from '../utils/azure';
import FileUploadForm from './FileUploadForm';
import FewShotForm from './FewShotForm';
import SearchForm from './SearchForm';
import { Message, AnalysisResult, ErrorResponse } from '../types';

const notyf = new Notyf();

const Chat: React.FC = () => {
  const { conversationId, setConversationId, messages, setMessages, totalTokensUsed, setTotalTokensUsed } = useContext(ConversationContext);
  const [userMessage, setUserMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(1);
  const [maxTokens, setMaxTokens] = useState(1000);
  const [activeFeature, setActiveFeature] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const maxContextTokens = 4000;
  const chatHistoryRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!conversationId) {
      startNewConversation();
    } else {
      loadConversation(conversationId);
    }
  }, [conversationId]);

  useEffect(() => {
    if (!conversationId) return;

    const pusher = new Pusher(process.env.NEXT_PUBLIC_PUSHER_KEY!, {
      cluster: process.env.NEXT_PUBLIC_PUSHER_CLUSTER!,
    });

    const channel = pusher.subscribe('chat-channel');
    channel.bind('new-message', (data: Message) => {
      if (data.id === conversationId) {
        setMessages((prevMessages) => [...prevMessages, data]);
        setIsTyping(false);
        setTotalTokensUsed((prevTokens) => prevTokens + countTokens(data.content));
      }
    });

    return () => {
      pusher.unsubscribe('chat-channel');
      pusher.disconnect();
    };
  }, [conversationId, setMessages, setTotalTokensUsed]);

  useEffect(() => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const toggleSidebar = () => {
    setIsSidebarOpen((prev) => !prev);
  };

  const sendMessage = async () => {
    if (!userMessage.trim() || !conversationId) return;

    const message = userMessage.trim();
    const newMessageTokens = countTokens(message);

    if (totalTokensUsed + newMessageTokens > maxContextTokens) {
      setError(`Message exceeds context window limit (${maxContextTokens} tokens). Please start a new conversation.`);
      return;
    }

    const newMessage: Message = {
      id: conversationId,
      sender: 'user',
      content: message,
      timestamp: Date.now()
    };

    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setUserMessage('');
    setIsTyping(true);
    setError(null);

    try {
      const response = await fetchWithAuth('/api/send_message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          conversation_id: conversationId, 
          message,
          temperature,
          top_p: topP,
          max_tokens: maxTokens,
        }),
      });

      const data: AnalysisResult | ErrorResponse = await response.json();

      if (response.ok) {
        setTotalTokensUsed((prevTokens) => prevTokens + newMessageTokens);
      } else {
        setError((data as ErrorResponse).error || 'Failed to send message. Please try again.');
        setIsTyping(false);
      }
    } catch (error: any) {
      setError(error.message || 'An unexpected error occurred. Please try again.');
      setIsTyping(false);
    }
  };

  const startNewConversation = async () => {
    try {
      const response = await fetchWithAuth('/api/start_conversation', { method: 'POST' });
      const data: { conversation_id: string } | ErrorResponse = await response.json();

      if (response.ok && 'conversation_id' in data) {
        setConversationId(data.conversation_id);
        setMessages([]);
        setTotalTokensUsed(0);
        notyf.success('Started a new conversation.');
        setIsSidebarOpen(false);
        setError(null);
      } else {
        setError((data as ErrorResponse).error || 'Failed to start a new conversation. Please try again.');
      }
    } catch (error: any) {
      setError(error.message || 'An unexpected error occurred while starting a new conversation.');
    }
  };

  const resetConversation = async () => {
    if (!conversationId) return;

    try {
      const response = await fetchWithAuth('/api/reset_conversation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ conversation_id: conversationId }),
      });

      const data: AnalysisResult | ErrorResponse = await response.json();

      if (response.ok) {
        setMessages([]);
        setTotalTokensUsed(0);
        notyf.success('Conversation has been reset.');
        setError(null);
      } else {
        setError((data as ErrorResponse).error || 'Failed to reset conversation. Please try again.');
      }
    } catch (error: any) {
      setError(error.message || 'An unexpected error occurred while resetting the conversation.');
    }
  };

  const loadConversation = async (convId: string) => {
    try {
      const response = await fetchWithAuth(`/api/load_conversation/${convId}`, { method: 'GET' });
      const data: { conversation: Message[] } | ErrorResponse = await response.json();

      if (response.ok && 'conversation' in data) {
        setMessages(data.conversation);
        setTotalTokensUsed(data.conversation.reduce((total, msg) => total + countTokens(msg.content), 0));
        notyf.success('Conversation loaded.');
        setIsSidebarOpen(false);
        setError(null);
      } else {
        setError((data as ErrorResponse).error || 'Failed to load conversation. Please try again.');
      }
    } catch (error: any) {
      setError(error.message || 'An unexpected error occurred while loading the conversation.');
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setUserMessage(e.target.value);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const renderActiveFeature = () => {
    switch (activeFeature) {
      case 'fileUpload':
        return <FileUploadForm />;
      case 'fewShot':
        return <FewShotForm />;
      case 'search':
        return <SearchForm />;
      default:
        return null;
    }
  };

  return (
    <div className={`chat-page ${isSidebarOpen ? 'sidebar-open' : ''}`}>
      <aside className={`conversation-sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <ConversationList />
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <h1>Chatbot</h1>
          <nav className="chat-nav">
            <button onClick={toggleSidebar} title="Toggle Conversation History" aria-label="Toggle Conversation History">
              <FontAwesomeIcon icon={faBars} />
            </button>
            <button onClick={startNewConversation} title="Start New Conversation" aria-label="Start New Conversation">
              <FontAwesomeIcon icon={faPlus} />
            </button>
            <button onClick={resetConversation} title="Reset Conversation" aria-label="Reset Conversation">
              <FontAwesomeIcon icon={faRedo} />
            </button>
            <button onClick={toggleSidebar} title="Conversation History" aria-label="Conversation History">
              <FontAwesomeIcon icon={faHistory} />
            </button>
          </nav>
        </header>

        <div className="chat-container">
          <div className="chat-content">
            <div className="chat-history" ref={chatHistoryRef}>
              {messages.map((msg, index) => (
                <div key={index} className={`message ${msg.sender}`}>
                  <div className="message-content">{msg.content}</div>
                </div>
              ))}
              {isTyping && (
                <div className="message bot">
                  <div className="message-content typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              )}
            </div>

            <div className="message-input">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  sendMessage();
                }}
              >
                <textarea
                  value={userMessage}
                  onChange={handleInput}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your message and press Enter..."
                  className="message-input-field"
                  rows={1}
                  aria-label="Message Input"
                />
                <button type="submit" className="send-button" aria-label="Send Message">
                  <FontAwesomeIcon icon={faPaperPlane} />
                </button>
              </form>

              {error && <div className="error-message">{error}</div>}

              <div className="model-params">
                <div>
                  <label htmlFor="temperature">Temperature ({temperature.toFixed(1)}):</label>
                  <input
                    type="range"
                    id="temperature"
                    min="0"
                    max="1"
                    step="0.1"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  />
                </div>
                <div>
                  <label htmlFor="topP">Top P ({topP.toFixed(1)}):</label>
                  <input
                    type="range"
                    id="topP"
                    min="0"
                    max="1"
                    step="0.1"
                    value={topP}
                    onChange={(e) => setTopP(parseFloat(e.target.value))}
                  />
                </div>
                <div>
                  <label htmlFor="maxTokens">Max Tokens ({maxTokens}):</label>
                  <input
                    type="number"
                    id="maxTokens"
                    min="1"
                    max="4000"
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(parseInt(e.target.value, 10) || 1)}
                  />
                </div>
              </div>

              <div className="token-usage">
                Tokens Used: {totalTokensUsed} / {maxContextTokens}
              </div>
            </div>
          </div>

          <div className="feature-sidebar">
            <button onClick={() => setActiveFeature('fileUpload')} title="File Upload" aria-label="File Upload">
              <FontAwesomeIcon icon={faUpload} />
            </button>
            <button onClick={() => setActiveFeature('fewShot')} title="Few-Shot Learning" aria-label="Few-Shot Learning">
              <FontAwesomeIcon icon={faLightbulb} />
            </button>
            <button onClick={() => setActiveFeature('search')} title="Search" aria-label="Search">
              <FontAwesomeIcon icon={faSearch} />
            </button>
          </div>

          {renderActiveFeature()}
        </div>
      </main>
    </div>
  );
};

export default Chat;
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/pages/index.tsx

## Summary

This JavaScript file contains the implementation of the HomePage component and an authentication verification function. It serves as a crucial part of the application's frontend, managing the user interface and ensuring that users are authenticated before accessing certain features.

## Changes Made

- Added HomePage component to manage the main user interface.
- Implemented verifyAuth function to check user authentication status.

```typescript
import React, { useEffect, useContext, useState } from 'react';
import { useRouter } from 'next/router';
import Chat from '../components/Chat';
import FewShotForm from '../components/FewShotForm';
import FileUploadForm from '../components/FileUploadForm';
import SearchForm from '../components/SearchForm';
import Header from '../components/Header';
import { ConversationContext } from '../contexts/ConversationContext';
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';
import './index.css'; // Create this CSS file for styling

const notyf = new Notyf();

const HomePage: React.FC = () => {
  const router = useRouter();
  const { setConversationId } = useContext(ConversationContext);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);

  useEffect(() => {
    const verifyAuth = async () => {
      try {
        const response = await fetch('/api/protected_route', { method: 'GET' });
        if (response.ok) {
          setIsAuthenticated(true);
        } else {
          notyf.error('Authentication required.');
          router.push('/login');
        }
      } catch (error: any) {
        notyf.error('Authentication failed.');
        router.push('/login');
      }
    };

    verifyAuth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!isAuthenticated) {
    return <div>Loading...</div>; // Optionally, render a loading spinner
  }

  return (
    <div className="home-page">
      <Header />
      <div className="main-content">
        <Chat />
        <aside className="side-panel">
          <FewShotForm />
          <FileUploadForm />
        </aside>
      </div>
    </div>
  );
};

export default HomePage;
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/pages/_document.tsx

## Summary

This JavaScript code defines a class named `MyDocument`, which extends the base class `Document`. It is designed to manage the document structure and rendering logic within a web application context, likely in a framework that supports server-side rendering or document manipulation.

## Changes Made

- Added MyDocument class that extends Document
- Implemented getInitialProps method for initial data fetching
- Implemented render method for rendering the document structure

```typescript
// File: apps/frontend/src/pages/_document.tsx

import Document, { Html, Head, Main, NextScript, DocumentContext } from 'next/document';
import { ServerStyleSheet } from 'styled-components'; // If using styled-components

class MyDocument extends Document {
  static async getInitialProps(ctx: DocumentContext) {
    // If using styled-components, integrate with ServerStyleSheet
    const sheet = new ServerStyleSheet();
    const originalRenderPage = ctx.renderPage;

    try {
      ctx.renderPage = () =>
        originalRenderPage({
          enhanceApp: (App) => (props) => sheet.collectStyles(<App {...props} />),
        });

      const initialProps = await Document.getInitialProps(ctx);
      return {
        ...initialProps,
        styles: (
          <>
            {initialProps.styles}
            {sheet.getStyleElement()}
          </>
        )
      };
    } finally {
      sheet.seal();
    }
  }

  render() {
    return (
      <Html lang="en">
        <Head>
          {/* Font Awesome CSS is already handled in _app.tsx via styled-components */}
          <link rel="icon" href="/favicon.ico" />
          {/* Add any additional head elements here */}
        </Head>
        <body>
          <Main />
          <NextScript />
        </body>
      </Html>
    );
  }
}

export default MyDocument;
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/components/Chat.css

## Summary

This CSS file is designed to style a chat application, providing a responsive and user-friendly interface. It includes styles for various components such as chat pages, message displays, input fields, and sidebars, ensuring a cohesive look and feel across different screen sizes and modes (light and dark).

## Changes Made

- Added styles for dark mode support.
- Updated flexbox layout for better responsiveness.
- Fixed button hover effects for improved user experience.

```css
/* File: apps/frontend/src/components/Chat.css */

/* Base styles */
/* Main container for the chat application, utilizing flexbox for layout. */
.chat-page {
  display: flex;
  height: 100vh;
  font-family: 'Helvetica Neue', Arial, sans-serif;
  background-color: #f5f5f5;
}

/* Container for the main chat content, allowing for flexible layout adjustments. */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  transition: margin-left 0.3s ease;
}

/* Header section of the chat, containing title and navigation elements. */
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 15px 20px;
  background-color: #ffffff;
  border-bottom: 1px solid #e0e0e0;
}

/* Title styling within the chat header. */
.chat-header h1 {
  margin: 0;
  font-size: 24px;
  color: #333333;
}

/* Styling for navigation buttons in the chat. */
.chat-nav button {
  background: none;
  border: none;
  margin-left: 15px;
  cursor: pointer;
}

/* Styling for SVG icons in the navigation. */
.chat-nav svg {
  font-size: 20px;
  color: #333333;
}

/* Hover effect for navigation SVG icons. */
.chat-nav svg:hover {
  color: #007bff;
}

/* Sidebar styles */
/* Sidebar for conversation list, initially hidden off-screen. */
.conversation-sidebar {
  position: fixed;
  left: -300px;
  top: 0;
  width: 300px;
  height: 100%;
  background-color: #ffffff;
  border-right: 1px solid #e0e0e0;
  overflow-y: auto;
  transition: left 0.3s ease;
  z-index: 1000;
}

/* Style to bring the sidebar into view when open. */
.conversation-sidebar.open {
  left: 0;
}

/* Adjusts main chat area margin when sidebar is open. */
.chat-main.sidebar-open {
  margin-left: 300px;
}

/* Chat container */
/* Container for chat messages and input area. */
.chat-container {
  flex: 1;
  display: flex;
  overflow: hidden;
}

/* Content area for chat messages. */
.chat-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Scrollable area for chat history. */
.chat-history {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
}

/* Container for individual messages. */
.message {
  display: flex;
  margin-bottom: 15px;
}

/* Styling for the content of messages. */
.message-content {
  max-width: 60%;
  padding: 15px;
  border-radius: 10px;
  font-size: 16px;
  line-height: 1.5;
}

/* Aligns user messages to the right. */
.message.user {
  justify-content: flex-end;
}

/* User message content styling. */
.message.user .message-content {
  background-color: #dcf8c6;
  color: #000000;
  border-top-right-radius: 0;
}

/* Aligns assistant messages to the left. */
.message.assistant {
  justify-content: flex-start;
}

/* Assistant message content styling. */
.message.assistant .message-content {
  background-color: #ffffff;
  color: #000000;
  border-top-left-radius: 0;
  border: 1px solid #e0e0e0;
}

/* Typing indicator */
/* Indicator for typing status. */
.typing-indicator {
  display: flex;
  align-items: center;
  justify-content: flex-start;
}

/* Styling for individual dots in the typing indicator. */
.typing-indicator span {
  display: inline-block;
  width: 8px;
  height: 8px;
  margin: 0 2px;
  background-color: #cccccc;
  border-radius: 50%;
  animation: typing 1s infinite;
}

/* Delay for the second dot in the typing indicator. */
.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

/* Delay for the third dot in the typing indicator. */
.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

/* Animation for the typing indicator dots. */
@keyframes typing {
  0% {
    opacity: 0.2;
    transform: scale(1);
  }
  20% {
    opacity: 1;
    transform: scale(1.3);
  }
  100% {
    opacity: 0.2;
    transform: scale(1);
  }
}

/* Message input area */
/* Input area for sending messages. */
.message-input {
  padding: 15px 20px;
  background-color: #ffffff;
  border-top: 1px solid #e0e0e0;
}

/* Form layout for message input. */
.message-input form {
  display: flex;
  align-items: center;
}

/* Text area for message input. */
.message-input-field {
  flex: 1;
  padding: 10px 15px;
  font-size: 16px;
  border: 1px solid #e0e0e0;
  border-radius: 20px;
  outline: none;
  resize: none;
  max-height: 150px;
  overflow-y: auto;
}

/* Focus effect for the message input field. */
.message-input-field:focus {
  border-color: #007bff;
}

/* Button for sending messages. */
.send-button {
  margin-left: 10px;
  padding: 10px 15px;
  font-size: 16px;
  background-color: #007bff;
  color: #ffffff;
  border: none;
  border-radius: 50%;
  cursor: pointer;
}

/* Styling for SVG icons in the send button. */
.send-button svg {
  font-size: 18px;
}

/* Hover effect for the send button. */
.send-button:hover {
  background-color: #0056b3;
}

/* Feature sidebar */
/* Sidebar for additional features. */
.feature-sidebar {
  width: 60px;
  background-color: #f0f0f0;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px 0;
}

/* Styling for buttons in the feature sidebar. */
.feature-sidebar button {
  background: none;
  border: none;
  margin-bottom: 20px;
  cursor: pointer;
  font-size: 24px;
  color: #333333;
}

/* Hover effect for feature sidebar buttons. */
.feature-sidebar button:hover {
  color: #007bff;
}

/* New feature components */
.file-upload-form,
.few-shot-form,
.search-form {
  width: 300px;
  padding: 20px;
  background-color: #ffffff;
  border-left: 1px solid #e0e0e0;
}

.file-upload-form h2,
.few-shot-form h2,
.search-form h2 {
  margin-top: 0;
  margin-bottom: 20px;
  font-size: 20px;
  color: #333333;
}

.file-upload-form input[type="file"],
.few-shot-form textarea,
.search-form input[type="text"] {
  width: 100%;
  padding: 10px;
  margin-bottom: 10px;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
}

.file-upload-form button,
.few-shot-form button,
.search-form button {
  width: 100%;
  padding: 10px;
  background-color: #007bff;
  color: #ffffff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.file-upload-form button:hover,
.few-shot-form button:hover,
.search-form button:hover {
  background-color: #0056b3;
}

/* Container for displaying search results. */
.search-results {
  margin-top: 20px;
}

/* Heading styles for search results. */
.search-results h3 {
  font-size: 18px;
  margin-bottom: 10px;
}

/* Styling for the search results list. */
.search-results ul {
  list-style-type: none;
  padding: 0;
}

/* Individual search result item styling. */
.search-results li {
  margin-bottom: 10px;
  padding: 10px;
  background-color: #f5f5f5;
  border-radius: 4px;
}

/* Scrollbar styling */
/* Scrollbar styling for chat history. */
.chat-history::-webkit-scrollbar {
  width: 8px;
}

/* Track styling for chat history scrollbar. */
.chat-history::-webkit-scrollbar-track {
  background: #f1f1f1;
}

/* Thumb styling for chat history scrollbar. */
.chat-history::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 4px;
}

/* Hover effect for chat history scrollbar thumb. */
.chat-history::-webkit-scrollbar-thumb:hover {
  background: #555;
}

/* Responsive adjustments */
/* Responsive adjustments for smaller screens. */
@media (max-width: 768px) {
/* Styling for the content of messages. */
  .message-content {
    max-width: 80%;
    font-size: 14px;
  }

/* Text area for message input. */
  .message-input-field {
    font-size: 14px;
  }

/* Button for sending messages. */
  .send-button {
    font-size: 14px;
    padding: 8px 12px;
  }

/* Sidebar for conversation list, initially hidden off-screen. */
  .conversation-sidebar {
    width: 250px;
    left: -250px;
  }

/* Style to bring the sidebar into view when open. */
  .conversation-sidebar.open {
    left: 0;
  }

/* Adjusts main chat area margin when sidebar is open. */
  .chat-main.sidebar-open {
    margin-left: 250px;
  }

/* Sidebar for additional features. */
  .feature-sidebar {
    width: 50px;
  }

  .file-upload-form,
  .few-shot-form,
  .search-form {
    width: 250px;
  }
}

/* Dark mode styles */
/* Dark mode styling for the chat page. */
.chat-page.dark-mode {
  background-color: #2c2c2c;
  color: #ffffff;
}

/* Dark mode styling for the chat header. */
.chat-page.dark-mode .chat-header {
  background-color: #1f1f1f;
  border-bottom-color: #444444;
}

/* Dark mode styling for assistant messages. */
.chat-page.dark-mode .message.assistant .message-content {
  background-color: #3a3a3a;
  border-color: #555555;
}

/* Dark mode styling for user messages. */
.chat-page.dark-mode .message.user .message-content {
  background-color: #4a4a4a;
}

/* Dark mode styling for the message input area. */
.chat-page.dark-mode .message-input {
  background-color: #1f1f1f;
  border-top-color: #444444;
}

/* Dark mode styling for the message input field. */
.chat-page.dark-mode .message-input-field {
  background-color: #3a3a3a;
  color: #ffffff;
  border-color: #555555;
}

/* Dark mode styling for the send button. */
.chat-page.dark-mode .send-button {
  background-color: #007bff;
}

/* Dark mode hover effect for the send button. */
.chat-page.dark-mode .send-button:hover {
  background-color: #0056b3;
}

/* Dark mode styling for the feature sidebar. */
.chat-page.dark-mode .feature-sidebar {
  background-color: #1f1f1f;
}

/* Dark mode styling for feature sidebar buttons. */
.chat-page.dark-mode .feature-sidebar button {
  color: #ffffff;
}

.chat-page.dark-mode .file-upload-form,
.chat-page.dark-mode .few-shot-form,
.chat-page.dark-mode .search-form {
  background-color: #1f1f1f;
  border-left-color: #444444;
}

.chat-page.dark-mode .file-upload-form input[type="file"],
.chat-page.dark-mode .few-shot-form textarea,
.chat-page.dark-mode .search-form input[type="text"] {
  background-color: #3a3a3a;
  color: #ffffff;
  border-color: #555555;
}

/* Dark mode styling for search result items. */
.chat-page.dark-mode .search-results li {
  background-color: #3a3a3a;
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/components/FileUploadForm.tsx

## Summary

This JavaScript code defines a file upload form component that allows users to select and submit files. It handles file selection and form submission events, providing a user-friendly interface for file uploads in a React application.

## Changes Made

- Added FileUploadForm component to handle file uploads.
- Implemented handleFileChange function to manage file input changes.
- Created handleSubmit function to process the form submission.

```typescript
import React, { useState } from 'react';
import fetchWithAuth from '../utils/fetchWithAuth';
import { Notyf } from 'notyf';

const notyf = new Notyf();

const FileUploadForm: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file to upload');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetchWithAuth('/api/upload_file', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        notyf.success(data.message || 'File uploaded successfully');
        setFile(null);
        setError(null);
      } else {
        setError(data.error || 'Failed to upload file');
        notyf.error(data.error || 'Failed to upload file');
      }
    } catch (error: any) {
      const errorMessage = error.message || 'An error occurred while uploading the file';
      setError(errorMessage);
      notyf.error(errorMessage);
    }
  };

  return (
    <div className="file-upload-form">
      <h2>Upload File</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit" disabled={!file}>
          Upload
        </button>
      </form>
      {error && <div className="error-message">{error}</div>}
    </div>
  );
};

export default FileUploadForm;
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/custom.d.ts

## Summary

This JavaScript code provides a CORS (Cross-Origin Resource Sharing) middleware function for handling CORS requests in a Next.js application. It allows the server to specify which origins are permitted to access resources, as well as the HTTP methods and headers that can be used in requests.

## Changes Made

- Added CORS middleware function to handle cross-origin requests.

```typescript
// apps/backend/custom.d.ts

declare module 'nextjs-cors' {
  import { NextApiRequest, NextApiResponse } from 'next';
  export default function cors(
    req: NextApiRequest,
    res: NextApiResponse,
    options: {
      origin?: string | string[];
      methods?: string | string[];
      allowedHeaders?: string | string[];
      exposedHeaders?: string | string[];
      credentials?: boolean;
      maxAge?: number;
    }
  ): Promise<void>;
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/utils/config.ts

## Summary

This JavaScript file serves as a foundational module for a larger application, providing essential functionality and structure. It is designed to be extended with additional functions and classes as the application grows, ensuring maintainability and scalability.

## Changes Made



```typescript
// File: apps/frontend/src/utils/config.ts

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:3000';
export const PUSHER_KEY = process.env.NEXT_PUBLIC_PUSHER_KEY || '';
export const PUSHER_CLUSTER = process.env.NEXT_PUBLIC_PUSHER_CLUSTER || '';
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/pages/register.tsx

## Summary

This JavaScript file contains the implementation of a registration page for a web application. It includes functions for rendering the registration form and handling the registration process when the form is submitted. The code is designed to integrate with a React application, allowing users to create new accounts by providing their information.

## Changes Made

- Added RegisterPage function to render the registration form.
- Implemented handleRegister function to manage form submission and user registration.

```typescript
import React, { useState } from 'react';
import { useRouter } from 'next/router';
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';
import { API_BASE_URL } from '../utils/config';
import './register.css'; // Create this CSS file for styling

const notyf = new Notyf();

const RegisterPage: React.FC = () => {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState<boolean>(false);

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!email.trim() || !password.trim() || !confirmPassword.trim()) {
      notyf.error('All fields are required.');
      return;
    }

    if (password !== confirmPassword) {
      notyf.error('Passwords do not match.');
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email: email.trim(), password: password.trim() }),
      });

      const data = await response.json();

      if (response.ok) {
        notyf.success('Registered successfully.');
        router.push('/'); // Redirect to home or chat page
      } else {
        notyf.error(data.message || 'Registration failed.');
      }
    } catch (error: any) {
      notyf.error(error.message || 'Registration failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="register-page">
      <form className="register-form" onSubmit={handleRegister}>
        <h1>Register</h1>
        <div className="form-group">
          <label htmlFor="email">Email:</label>
          <input
            type="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email"
            aria-label="Email"
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="password">Password:</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter your password"
            aria-label="Password"
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="confirm-password">Confirm Password:</label>
          <input
            type="password"
            id="confirm-password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            placeholder="Confirm your password"
            aria-label="Confirm Password"
            required
          />
        </div>
        <button type="submit" className="btn btn-register" disabled={loading}>
          {loading ? 'Registering...' : 'Register'}
        </button>
        <p className="redirect-text">
          Already have an account? <a href="/login">Login here</a>
        </p>
      </form>
    </div>
  );
};

export default RegisterPage;
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/components/ConversationList.tsx

## Summary

This JavaScript file is responsible for managing and displaying a list of conversations in a chat application. It includes functions to fetch conversations from a server, handle user interactions with the conversation list, and render the conversation items dynamically.

## Changes Made

- Added ConversationList function to render conversation items.
- Implemented fetchConversations function to retrieve conversation data from the server.
- Created handleConversationClick function to manage user clicks on conversation items.

```typescript
// File: chatapp-vercel/apps/frontend/src/components/ConversationList.tsx

import React, { useState, useEffect, useContext } from 'react';
import { ConversationContext } from '../contexts/ConversationContext';
import fetchWithAuth from '../utils/fetchWithAuth';
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';
import { Conversation, ApiResponse } from '../utils/types';
import './ConversationList.css';

const notyf = new Notyf();

const ConversationList: React.FC = () => {
  const { setConversationId } = useContext(ConversationContext);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchConversations();
  }, []);

  const fetchConversations = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetchWithAuth('/api/list_conversations', {
        method: 'GET',
      });
      const data: ApiResponse<{ conversations: Conversation[] }> = await response.json();
      
      if (response.ok) {
        setConversations(data.conversations || []);
      } else {
        throw new Error(data.error || 'Failed to load conversations');
      }
    } catch (error: any) {
      setError(error.message || 'An unexpected error occurred while loading conversations');
      notyf.error(error.message || 'Failed to load conversations. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleConversationClick = async (conversation_id: string) => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetchWithAuth(`/api/load_conversation/${conversation_id}`, {
        method: 'GET',
      });
      const data = await response.json();

      if (response.ok) {
        setConversationId(conversation_id);
        notyf.success('Conversation loaded successfully.');
      } else {
        throw new Error(data.error || 'Failed to load conversation');
      }
    } catch (error: any) {
      setError(error.message || 'An unexpected error occurred while loading the conversation');
      notyf.error(error.message || 'Failed to load conversation. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="conversation-list">Loading conversations...</div>;
  }

  if (error) {
    return (
      <div className="conversation-list">
        <div className="error-message">{error}</div>
        <button onClick={fetchConversations}>Retry</button>
      </div>
    );
  }

  return (
    <div className="conversation-list">
      <h2>Conversations</h2>
      {conversations.length === 0 ? (
        <p>No conversations found. Start a new one!</p>
      ) : (
        <ul>
          {conversations.map((conv) => (
            <li key={conv.conversation_id}>
              <button onClick={() => handleConversationClick(conv.conversation_id)}>
                {conv.title || 'Untitled Conversation'}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default ConversationList;
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/utils/auth.ts

## Summary

This JavaScript module provides utility functions for managing user authentication tokens. It includes methods to save, retrieve, and remove tokens, as well as to get user information associated with the tokens. This functionality is essential for maintaining user sessions in web applications.

## Changes Made

- Added saveToken function to store authentication tokens.
- Added getToken function to retrieve stored tokens.
- Added removeToken function to delete tokens.
- Added getUser function to fetch user information based on the token.

```typescript
import jwtDecode from 'jwt-decode';

interface User {
  id: string;
  email: string;
}

export function saveToken(token: string): void {
  localStorage.setItem('jwt_token', token);
}

export function getToken(): string | null {
  return localStorage.getItem('jwt_token');
}

export function removeToken(): void {
  localStorage.removeItem('jwt_token');
}

export function getUser(): User | null {
  const token = getToken();
  if (token) {
    try {
      const user = jwtDecode<User>(token);
      return user;
    } catch (error) {
      console.error('Invalid token:', error);
      removeToken();
    }
  }
  return null;
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/components/LogoutButton.tsx

## Summary

This JavaScript file contains functions related to user logout functionality within a web application. It provides the necessary components to handle user logout actions effectively, ensuring that user sessions are terminated securely and appropriately.

## Changes Made

- Added LogoutButton function to create a logout button component.
- Implemented handleLogout function to manage the logout process.

```typescript
import React from 'react';
import { Notyf } from 'notyf';
import 'notyf/notyf.min.css';
import { API_BASE_URL } from '../utils/config';
import { useRouter } from 'next/router';
import './LogoutButton.css'; // Create this CSS file for styling

const notyf = new Notyf();

const LogoutButton: React.FC = () => {
  const router = useRouter();

  const handleLogout = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/logout`, {
        method: 'POST',
      });

      const data = await response.json();

      if (response.ok) {
        notyf.success('Logged out successfully.');
        router.push('/login'); // Redirect to login page
      } else {
        notyf.error(data.message || 'Logout failed.');
      }
    } catch (error: any) {
      notyf.error(error.message || 'Logout failed.');
    }
  };

  return (
    <button className="btn btn-logout" onClick={handleLogout}>
      Logout
    </button>
  );
};

export default LogoutButton;
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/pages/api/register.ts

## Summary

This JavaScript file contains an API handler function designed to process registration requests. It serves as part of a larger application that manages user registrations, ensuring that incoming requests are handled appropriately and responses are sent back to the client.

## Changes Made

- Added the handler function to process registration requests.

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import bcrypt from 'bcrypt';
import clientPromise from '@/utils/mongodb';
import { generateToken, setTokenCookie } from '@/utils/auth';
import { apiHandler } from '@/utils/apiHandler';

interface RegisterRequest extends NextApiRequest {
  body: {
    email: string;
    password: string;
  };
}

interface RegisterResponse {
  message: string;
}

const handler = async (req: RegisterRequest, res: NextApiResponse<RegisterResponse>) => {
  if (req.method !== 'POST') {
    throw { statusCode: 405, message: `Method ${req.method} Not Allowed` };
  }

  const { email, password } = req.body;

  if (!email || !password) {
    throw { statusCode: 400, message: 'Email and password are required.' };
  }

  const client = await clientPromise;
  const db = client.db(process.env.MONGODB_DB_NAME);
  const users = db.collection('users');

  const existingUser = await users.findOne({ email: email.toLowerCase() });
  if (existingUser) {
    throw { statusCode: 409, message: 'Email is already registered.' };
  }

  const saltRounds = 10;
  const hashedPassword = await bcrypt.hash(password, saltRounds);

  const newUser = await users.insertOne({
    email: email.toLowerCase(),
    passwordHash: hashedPassword,
    createdAt: new Date(),
  });

  const token = generateToken({ id: newUser.insertedId.toString(), email: email.toLowerCase() });
  setTokenCookie(res, token);

  res.status(201).json({ message: 'User registered successfully.' });
};

export default apiHandler(handler);
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/pages/_app.tsx

## Summary

This JavaScript file serves as the main application component for a web application, handling authentication and conversation loading functionalities. It integrates various components and manages the application state, ensuring a seamless user experience.

## Changes Made

- Added MyApp function to initialize the application.
- Updated checkAuth function to improve authentication logic.
- Fixed handleLoadConversation function to correctly handle event parameters.

```typescript
// File: apps/frontend/src/pages/_app.tsx

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import type { AppProps } from 'next/app';
import '../styles/globals.css';
import { library } from '@fortawesome/fontawesome-svg-core';
import { faBars, faPlus, faRedo, faPaperPlane, faHistory, faSearch } from '@fortawesome/free-solid-svg-icons';
import { config } from '@fortawesome/fontawesome-svg-core';
import '@fortawesome/fontawesome-svg-core/styles.css';
import { ConversationProvider } from '../contexts/ConversationContext';
import Header from '../components/Header';

config.autoAddCss = false;

library.add(faBars, faPlus, faRedo, faPaperPlane, faHistory, faSearch);

const MyApp = ({ Component, pageProps }: AppProps) => {
  const router = useRouter();
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await fetch('/api/protected_route', { method: 'GET' });
        if (response.ok) {
          setIsAuthenticated(true);
        } else {
          setIsAuthenticated(false);
          if (router.pathname !== '/login' && router.pathname !== '/register') {
            router.push('/login');
          }
        }
      } catch {
        setIsAuthenticated(false);
        if (router.pathname !== '/login' && router.pathname !== '/register') {
          router.push('/login');
        }
      }
    };

    checkAuth();
  }, [router]);

  useEffect(() => {
    const handleLoadConversation = (e: Event) => {
      const customEvent = e as CustomEvent;
      const conversation_id = customEvent.detail;
      window.location.href = `/?conversation_id=${conversation_id}`;
    };

    window.addEventListener('loadConversation', handleLoadConversation);

    return () => {
      window.removeEventListener('loadConversation', handleLoadConversation);
    };
  }, []);

  if (isAuthenticated === null) {
    return <div>Loading...</div>;
  }

  return (
    <ConversationProvider>
      <Header />
      {isAuthenticated || router.pathname === '/login' || router.pathname === '/register' ? (
        <Component {...pageProps} />
      ) : null}
    </ConversationProvider>
  );
};

export default MyApp;
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/types/index.d.ts

## Summary

This JavaScript file serves as a foundational module for the application, providing essential functionality and structure for future development. It is designed to be extensible, allowing developers to add functions and classes as needed to enhance the application's capabilities.

## Changes Made



```typescript
export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export interface Conversation {
  conversation_id: string;
  user_id: string;
  messages: Message[];
  created_at: Date;
  updated_at: Date;
  title?: string;
}

export interface FewShotExample {
  user_id: string;
  user_prompt: string;
  assistant_response: string;
  created_at: Date;
}

export interface User {
  id: string;
  email: string;
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/pages/api/load_conversation/[conversation_id].ts

## Summary

This JavaScript file defines an API handler function for processing incoming requests and sending responses in a Next.js application. It serves as a central point for handling specific API routes, managing request data, and returning appropriate responses to the client.

## Changes Made

- Added the handler function to process API requests.
- Updated the function parameters to include NextApiRequest and NextApiResponse types.

```typescript
// File: apps/backend/pages/api/load_conversation/[conversation_id].ts

import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';
import { loadConversation } from '@/utils/conversation'; // Import the consolidated function
import { errorHandler } from '@/middleware/errorHandler'; // Import errorHandler

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  const user = authenticate(req, res);
  if (!user) return;

  const { conversation_id } = req.query;

  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    res.status(405).end(`Method ${req.method} Not Allowed`);
    return;
  }

  if (!conversation_id || typeof conversation_id !== 'string') {
    return res.status(400).json({ message: 'Conversation ID is required.' });
  }

  try {
    const messages = await loadConversation(conversation_id, user.id); // Use the consolidated function

    if (messages) {
      res.status(200).json({ conversation: messages });
    } else {
      const error = new Error('Conversation not found.');
      (error as any).status = 404;
      throw error; // Let errorHandler handle this
    }
  } catch (error) {
    // Let the errorHandler handle any errors from loadConversation
    throw error; 
  }
};

export default errorHandler(handler); // Apply errorHandler
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/contexts/ConversationContext.tsx

## Summary

This JavaScript code defines a `ConversationProvider` function component that serves as a context provider for managing conversation state in a React application. It encapsulates the logic for providing conversation-related data to its child components, allowing them to access and manipulate the conversation state easily.

## Changes Made

- Added ConversationProvider function to manage conversation state context.

```typescript
import React, { createContext, useState, ReactNode } from 'react';

interface MessageType {
  role: string;
  content: string;
}

interface ConversationContextProps {
  conversationId: string;
  setConversationId: (id: string) => void;
  messages: MessageType[];
  setMessages: (messages: MessageType[]) => void;
}

export const ConversationContext = createContext<ConversationContextProps | null>(null);

export const ConversationProvider = ({ children }: { children: ReactNode }) => {
  const [conversationId, setConversationId] = useState<string>('');
  const [messages, setMessages] = useState<MessageType[]>([]);

  return (
    <ConversationContext.Provider
      value={{ conversationId, setConversationId, messages, setMessages }}
    >
      {children}
    </ConversationContext.Provider>
  );
};
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/next-env.d.ts

## Summary

This JavaScript file serves as a foundational module for a larger application, providing essential functionalities and structures that can be utilized by other components. It is designed to be extensible and maintainable, ensuring that future enhancements can be integrated seamlessly.

## Changes Made



```typescript
/// <reference types="next" />
/// <reference types="next/image-types/global" />

// NOTE: This file should not be edited
// see https://nextjs.org/docs/basic-features/typescript for more information.
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/utils/pusherServer.ts

## Summary

This JavaScript file serves as a foundational module for the application, currently containing no functions or classes. It is designed to be extended in the future with additional functionality as the application evolves.

## Changes Made



```typescript
import Pusher from 'pusher';

const pusherServer = new Pusher({
  appId: process.env.PUSHER_APP_ID!,
  key: process.env.PUSHER_KEY!,
  secret: process.env.PUSHER_SECRET!,
  cluster: process.env.PUSHER_CLUSTER!,
  useTLS: true,
});

export default pusherServer;
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/utils/types.ts

## Summary

This JavaScript file serves as a foundational module for a larger application, providing essential functionality and structure. It is designed to be extended with additional functions and classes as the application grows.

## Changes Made



```typescript
// File: apps/frontend/src/utils/types.ts

export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export interface Conversation {
  conversation_id: string;
  title?: string;
  updated_at: string;
}

export interface FewShotExample {
  user_prompt: string;
  assistant_response: string;
}

export interface ApiResponse<T> {
  message?: string;
  token?: string;
  conversation_id?: string;
  conversations?: Conversation[];
  conversation?: Message[];
  analysis?: string;
  results?: Conversation[];
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/src/utils/fetchWithAuth.ts

## Summary

This JavaScript code provides a utility function for making authenticated HTTP requests. It simplifies the process of fetching resources from a server while ensuring that the necessary authentication headers are included in the request.

## Changes Made

- Added fetchWithAuth function for making authenticated requests.

```typescript
// File: apps/frontend/src/utils/fetchWithAuth.ts

import config from '../config';
import { getToken } from './auth';

export const fetchWithAuth = async <T = any>(url: string, options: RequestInit = {}): Promise<T> => {
  const token = getToken();

  const headers = {
    ...(options.headers || {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };

  const response = await fetch(`${config.API_BASE_URL}${url}`, {
    ...options,
    headers: {
      ...headers,
      ...(options.body && !(options.body instanceof FormData)
        ? { 'Content-Type': 'application/json' }
        : {}),
    },
  });

  const contentType = response.headers.get('content-type');
  let data: any;
  if (contentType && contentType.includes('application/json')) {
    data = await response.json();
  } else {
    data = await response.text();
  }

  if (!response.ok) {
    const error = data?.message || `HTTP error! Status: ${response.status}`;
    throw new Error(error);
  }

  return data;
};

export default fetchWithAuth;
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/next.config.js

## Summary

This JavaScript file serves as a foundational module for the application, designed to encapsulate core functionalities and provide a structure for future development. It currently contains no defined functions or classes, indicating that it may be a placeholder or a starting point for further implementation.

## Changes Made



```javascript
/** @type {import('next').NextConfig} */
module.exports = {
    typescript: {
      ignoreBuildErrors: false,
    },
  };
```

# File: /mnt/disks/data/chatapp-vercel/apps/frontend/next.config.js

## Summary

This JavaScript file serves as a foundational module for the application, providing essential utilities and structures that can be utilized across various components. It is designed to enhance code reusability and maintainability within the larger application context.

## Changes Made



```javascript
/** @type {import('next').NextConfig} */
module.exports = {
  reactStrictMode: true,
};
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/pages/api/get_config.ts

## Summary

This JavaScript file contains an API handler function designed to process incoming requests and send appropriate responses. It is part of a larger application that utilizes Next.js for server-side rendering and API routes.

## Changes Made

- Added API handler function for processing requests and responses.

```typescript
// File: apps/backend/pages/api/get_config.ts

import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';
import { apiHandler } from '@/utils/apiHandler';

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  const user = authenticate(req, res);
  if (!user) {
    throw { statusCode: 401, message: 'Unauthorized' };
  }

  if (req.method !== 'GET') {
    throw { statusCode: 405, message: `Method ${req.method} Not Allowed` };
  }

  const config = {
    max_tokens: process.env.MAX_TOKENS || '128000',
    reply_tokens: process.env.REPLY_TOKENS || '800',
    chunk_size_tokens: process.env.CHUNK_SIZE_TOKENS || '1000',
  };

  res.status(200).json({ config });
};

export default apiHandler(handler);
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/pages/api/search_conversations.ts

## Summary

This JavaScript file contains an API handler function designed to process incoming requests and send appropriate responses in a Next.js application. It serves as a middleware for handling API requests, ensuring that the application can respond to client interactions effectively.

## Changes Made

- Added handler function to manage API requests.

```typescript
// File: apps/backend/pages/api/search_conversations.ts

import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';
import clientPromise from '@/utils/mongodb';
import { apiHandler } from '@/utils/apiHandler';

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  const user = authenticate(req, res);
  if (!user) {
    throw { statusCode: 401, message: 'Unauthorized' };
  }

  if (req.method !== 'POST') {
    throw { statusCode: 405, message: `Method ${req.method} Not Allowed` };
  }

  const { query } = req.body;

  if (!query) {
    throw { statusCode: 400, message: 'Search query is required.' };
  }

  const client = await clientPromise;
  const db = client.db(process.env.MONGODB_DB_NAME);
  const conversations = db.collection('conversations');

  const results = await conversations
    .find({
      user_id: user.id,
      'messages.content': { $regex: query, $options: 'i' },
    })
    .project({ conversation_id: 1, title: 1, updated_at: 1 })
    .toArray();

  res.status(200).json({ results });
};

export default apiHandler(handler);
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/types/models.ts

## Summary

This JavaScript file serves as a foundational structure for a larger application, currently containing no defined functions or classes. It is intended to be expanded with additional functionality as the application develops.

## Changes Made



```typescript
// File: apps/backend/types/models.ts

export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export interface Conversation {
  conversation_id: string;
  user_id: string;
  messages: Message[];
  created_at: Date;
  updated_at: Date;
  title?: string;
}

export interface User {
  id: string;
  email: string;
  passwordHash?: string;
  created_at?: Date;
}

export interface FewShotExample {
  user_id: string;
  user_prompt: string;
  assistant_response: string;
  created_at: Date;
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/utils/mongodb.ts

## Summary

This JavaScript file serves as a foundational module for the application, currently containing no functions or classes. It is designed to be extended in the future with additional functionality as the application grows.

## Changes Made



```typescript
// File: apps/backend/utils/mongodb.ts

import { MongoClient } from 'mongodb';

const uri = process.env.MONGODB_URI || '';
const options = {};

if (!uri) {
  throw new Error('Please add your MongoDB URI to .env.local');
}

let client: MongoClient;
let clientPromise: Promise<MongoClient>;

declare global {
  // eslint-disable-next-line no-var
  var _mongoClientPromise: Promise<MongoClient>;
}

if (process.env.NODE_ENV === 'development') {
  if (!global._mongoClientPromise) {
    client = new MongoClient(uri, options);
    global._mongoClientPromise = client.connect();
  }
  clientPromise = global._mongoClientPromise;
} else {
  client = new MongoClient(uri, options);
  clientPromise = client.connect();
}

export default clientPromise;
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/utils/pusher.ts

## Summary

This JavaScript file serves as a foundational module for a larger application, providing essential functionalities and structures that can be utilized by other components. It is designed to be modular and reusable, ensuring that the code adheres to best practices in software development.

## Changes Made



```typescript
// File: apps/backend/utils/pusher.ts

import Pusher from 'pusher';

export const PusherInstance = new Pusher({
  appId: process.env.PUSHER_APP_ID || '',
  key: process.env.PUSHER_KEY || '',
  secret: process.env.PUSHER_SECRET || '',
  cluster: process.env.PUSHER_CLUSTER || '',
  useTLS: true,
});
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/utils/conversation.ts

## Summary

This JavaScript module is responsible for managing conversations within a chat application. It provides functions to start, reset, and load conversations based on user and conversation IDs. The code is designed to handle asynchronous operations, returning promises for each function to ensure smooth interaction with the backend services.

## Changes Made

- Added startConversation function to initiate a new conversation.
- Added resetConversation function to reset an existing conversation.
- Added loadConversation function to retrieve messages from a conversation.

```typescript
//apps/backend/utils/conversation.ts
import { v4 as uuidv4 } from 'uuid';
import clientPromise from './mongodb';
import { Conversation, Message } from '../types/models';

export const startConversation = async (userId: string): Promise<string> => {
  const client = await clientPromise;
  const db = client.db(process.env.MONGODB_DB_NAME);
  const conversations = db.collection<Conversation>('conversations');

  const conversationId = uuidv4();
  await conversations.insertOne({
    conversation_id: conversationId,
    user_id: userId,
    messages: [],
    created_at: new Date(),
    updated_at: new Date(),
  });

  return conversationId;
};

export const resetConversation = async (conversationId: string, userId: string): Promise<boolean> => {
  const client = await clientPromise;
  const db = client.db(process.env.MONGODB_DB_NAME);
  const conversations = db.collection<Conversation>('conversations');

  const result = await conversations.updateOne(
    { conversation_id: conversationId, user_id: userId },
    { $set: { messages: [], updated_at: new Date() } }
  );

  return result.modifiedCount === 1;
};

export const loadConversation = async (conversationId: string, userId: string): Promise<Message[] | null> => {
  const client = await clientPromise;
  const db = client.db(process.env.MONGODB_DB_NAME);
  const conversations = db.collection<Conversation>('conversations');

  const conversation = await conversations.findOne({
    conversation_id: conversationId,
    user_id: userId,
  });

  return conversation ? conversation.messages : null;
};
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/pages/api/start_conversation.ts

## Summary

This JavaScript code defines an API handler function for processing incoming requests and sending responses in a Next.js application. It serves as a central point for managing API interactions, ensuring that requests are handled appropriately and responses are formatted correctly.

## Changes Made

- Added API handler function to manage requests and responses.

```typescript
// File: apps/backend/pages/api/start_conversation.ts

import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';
import { startConversation } from '@/utils/conversation';
import { apiHandler } from '@/utils/apiHandler';

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  try {
    const user = authenticate(req, res);
    if (!user) {
      throw { statusCode: 401, message: 'Unauthorized: Invalid or missing authentication token.' };
    }

    if (req.method !== 'POST') {
      throw { statusCode: 405, message: `Method ${req.method} Not Allowed. Use POST to start a new conversation.` };
    }

    const conversationId = await startConversation(user.id);
    if (!conversationId) {
      throw { statusCode: 500, message: 'Failed to start a new conversation. Please try again.' };
    }

    res.status(200).json({ conversation_id: conversationId, message: 'New conversation started successfully.' });
  } catch (error: any) {
    console.error('Start Conversation Error:', error);
    if (error.statusCode) {
      throw error;
    } else {
      throw { statusCode: 500, message: `Internal Server Error: ${error.message || 'An unexpected error occurred while starting a new conversation.'}` };
    }
  }
};

export default apiHandler(handler);
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/utils/auth.ts

## Summary

This JavaScript module provides functions for handling user authentication tokens in a Next.js application. It includes functionalities for generating tokens, setting and removing cookies, and retrieving tokens from cookies. These utilities are essential for managing user sessions and ensuring secure access to protected resources within the application.

## Changes Made

- Added functions for token generation and management
- Updated parameter types for better clarity
- Fixed return types for consistency

```typescript
import jwt from 'jsonwebtoken';
import { NextApiRequest, NextApiResponse } from 'next';
import { serialize, parse } from 'cookie';

interface UserPayload {
  id: string;
  email: string;
}

const JWT_SECRET = process.env.JWT_SECRET || 'default_jwt_secret';
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '7d';
const COOKIE_NAME = process.env.COOKIE_NAME || 'chatapp_token';
const COOKIE_EXPIRES_IN = parseInt(process.env.COOKIE_EXPIRES_IN || '7', 10);
const COOKIE_SECURE = process.env.COOKIE_SECURE === 'true';

export function generateToken(user: UserPayload): string {
  return jwt.sign(user, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });
}

export function setTokenCookie(res: NextApiResponse, token: string): void {
  const cookie = serialize(COOKIE_NAME, token, {
    httpOnly: true,
    secure: COOKIE_SECURE, // Set to true in production
    sameSite: 'strict',
    path: '/',
    maxAge: COOKIE_EXPIRES_IN * 24 * 60 * 60, // Convert days to seconds
  });

  res.setHeader('Set-Cookie', cookie);
}

export function removeTokenCookie(res: NextApiResponse): void {
  const cookie = serialize(COOKIE_NAME, '', {
    httpOnly: true,
    secure: COOKIE_SECURE, // Set to true in production
    sameSite: 'strict',
    path: '/',
    expires: new Date(0),
  });

  res.setHeader('Set-Cookie', cookie);
}

export function getTokenFromCookie(req: NextApiRequest): string | null {
  const cookies = req.headers.cookie;
  if (!cookies) return null;

  const parsedCookies = parse(cookies);
  return parsedCookies[COOKIE_NAME] || null;
}

export function authenticate(req: NextApiRequest, res: NextApiResponse): UserPayload | null {
  const token = getTokenFromCookie(req);
  if (!token) {
    res.status(401).json({ message: 'Authentication token missing.' });
    return null;
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET) as UserPayload;
    return decoded;
  } catch (error) {
    res.status(401).json({ message: 'Invalid or expired authentication token.' });
    return null;
  }
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/pages/api/send_message.ts

## Summary

This JavaScript file contains an API handler function designed to process incoming requests and send appropriate responses in a Next.js application. It serves as a crucial component for handling server-side logic and interactions with the client.

## Changes Made

- Added handler function to process API requests.

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';
import clientPromise from '@/utils/mongodb';
import { getAzureResponse, countTokens, manageContextWindow } from '@/utils/azure';
import { PusherInstance } from '@/utils/pusher';
import { apiHandler } from '@/utils/apiHandler';

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  const user = authenticate(req, res);
  if (!user) {
    throw { statusCode: 401, message: 'Unauthorized: Invalid or missing authentication token.' };
  }

  if (req.method !== 'POST') {
    throw { statusCode: 405, message: `Method ${req.method} Not Allowed. Use POST to send messages.` };
  }

  const { conversation_id, message } = req.body;

  if (!conversation_id || !message) {
    throw { statusCode: 400, message: 'Bad Request: Conversation ID and message are required.' };
  }

  try {
    const client = await clientPromise;
    const db = client.db(process.env.MONGODB_DB_NAME);
    const conversations = db.collection('conversations');
    const fewShotExamples = db.collection('few_shot_examples');

    const conversation = await conversations.findOne({
      conversation_id,
      user_id: user.id,
    });

    if (!conversation) {
      throw { statusCode: 404, message: 'Not Found: The specified conversation does not exist or you do not have access to it.' };
    }

    // Retrieve few-shot examples
    const examples = await fewShotExamples
      .find({ user_id: user.id })
      .project({ user_prompt: 1, assistant_response: 1, _id: 0 })
      .toArray();

    // Construct the prompt with few-shot examples
    const promptMessages = examples.map((ex) => ({
      role: 'system',
      content: `User: ${ex.user_prompt}\nAssistant: ${ex.assistant_response}`,
    }));

    // Manage the context window (e.g., keep only the last 4000 tokens)
    const maxContextTokens = 4000; // Adjust this based on your model's context window size
    let contextWindow = manageContextWindow([...promptMessages, ...conversation.messages], maxContextTokens);

    // Add the current user message to the context window
    contextWindow.push({ role: 'user', content: message });

    // Calculate remaining tokens for the response
    const remainingTokens = maxContextTokens - countTokens(contextWindow.map(m => m.content).join(' '));

    // Get assistant's response from Azure OpenAI API
    const assistantResponse = await getAzureResponse(contextWindow, remainingTokens);

    // Add user's message to the conversation
    await conversations.updateOne(
      { conversation_id },
      {
        $push: {
          messages: { role: 'user', content: message, timestamp: new Date() },
        },
        $set: { updated_at: new Date() },
      }
    );

    // Add assistant's response to the conversation
    await conversations.updateOne(
      { conversation_id },
      {
        $push: {
          messages: { role: 'assistant', content: assistantResponse, timestamp: new Date() },
        },
        $set: { updated_at: new Date() },
      }
    );

    // Emit the messages via Pusher
    await PusherInstance.trigger('chat-channel', 'new-message', {
      conversation_id,
      role: 'assistant',
      content: assistantResponse,
    });

    res.status(200).json({ message: 'Message sent successfully.' });
  } catch (error: any) {
    console.error('Send Message Error:', error);
    if (error.statusCode) {
      throw error;
    } else {
      throw { statusCode: 500, message: `Internal Server Error: ${error.message || 'An unexpected error occurred while sending the message.'}` };
    }
  }
};

export default apiHandler(handler);
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/pages/api/add_few_shot_example.ts

## Summary

This JavaScript file contains an API handler function designed to process requests related to adding few-shot learning data. It serves as a backend endpoint for handling specific data submissions and responses within a larger application focused on machine learning or AI functionalities.

## Changes Made

- Added handler function for processing few-shot learning requests.

```typescript
// File: apps/backend/pages/api/add_few_shot_example.ts

import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';
import clientPromise from '@/utils/mongodb';
import { apiHandler } from '@/utils/apiHandler';

interface AddFewShotRequest extends NextApiRequest {
  body: {
    user_prompt: string;
    assistant_response: string;
  };
}

const handler = async (req: AddFewShotRequest, res: NextApiResponse) => {
  const user = authenticate(req, res);
  if (!user) {
    throw { statusCode: 401, message: 'Unauthorized' };
  }

  if (req.method !== 'POST') {
    throw { statusCode: 405, message: `Method ${req.method} Not Allowed` };
  }

  const { user_prompt, assistant_response } = req.body;

  if (!user_prompt || !assistant_response) {
    throw { statusCode: 400, message: 'User prompt and assistant response are required.' };
  }

  const client = await clientPromise;
  const db = client.db(process.env.MONGODB_DB_NAME);
  const fewShotExamples = db.collection('few_shot_examples');

  await fewShotExamples.insertOne({
    user_id: user.id,
    user_prompt,
    assistant_response,
    created_at: new Date(),
  });

  res.status(200).json({ message: 'Few-shot example added successfully.' });
};

export default apiHandler(handler);
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/middleware/cors.ts

## Summary

This JavaScript file contains a CORS middleware function designed for use in a Next.js application. It handles Cross-Origin Resource Sharing (CORS) requests, allowing the server to specify which domains are permitted to access its resources. This is essential for enabling secure API interactions between different origins.

## Changes Made

- Added CORS middleware function to handle cross-origin requests.

```typescript
// File: apps/backend/middleware/cors.ts

import { NextApiRequest, NextApiResponse } from 'next';
import NextCors from 'nextjs-cors';

export async function cors(req: NextApiRequest, res: NextApiResponse) {
  await NextCors(req, res, {
    origin: (process.env.ALLOWED_ORIGINS || '*').split(','),
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    optionsSuccessStatus: 200,
  });
}
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/pages/api/login.ts

## Summary

This JavaScript file contains an API handler function designed to process login requests. It interacts with the Next.js API framework to handle user authentication and respond with appropriate login responses.

## Changes Made

- Added handler function for processing login requests.

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import bcrypt from 'bcrypt';
import clientPromise from '@/utils/mongodb';
import { generateToken, setTokenCookie } from '@/utils/auth';
import { apiHandler } from '@/utils/apiHandler';

interface LoginRequest extends NextApiRequest {
  body: {
    email: string;
    password: string;
  };
}

interface LoginResponse {
  message: string;
}

const handler = async (req: LoginRequest, res: NextApiResponse<LoginResponse>) => {
  if (req.method !== 'POST') {
    throw { statusCode: 405, message: `Method ${req.method} Not Allowed` };
  }

  const { email, password } = req.body;

  if (!email || !password) {
    throw { statusCode: 400, message: 'Email and password are required.' };
  }

  const client = await clientPromise;
  const db = client.db(process.env.MONGODB_DB_NAME);
  const users = db.collection('users');

  const user = await users.findOne({ email: email.toLowerCase() });
  if (!user) {
    throw { statusCode: 401, message: 'Invalid email or password.' };
  }

  const isPasswordValid = await bcrypt.compare(password, user.passwordHash);
  if (!isPasswordValid) {
    throw { statusCode: 401, message: 'Invalid email or password.' };
  }

  const token = generateToken({ id: user._id.toString(), email: email.toLowerCase() });
  setTokenCookie(res, token);

  res.status(200).json({ message: 'Logged in successfully.' });
};

export default apiHandler(handler);
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/middleware/auth.ts

## Summary

This JavaScript code provides an authentication middleware function for a Next.js API route. It ensures that incoming requests are properly authenticated before proceeding to the next middleware or route handler, thereby enhancing the security of the application.

## Changes Made

- Added authentication middleware function 'authenticate'.

```typescript
import { NextApiRequest, NextApiResponse } from 'next';
import { verifyToken } from '@/utils/auth';

export const authenticate = async (
  req: NextApiRequest,
  res: NextApiResponse,
  next: () => void
) => {
  try {
    const token = req.cookies.token;

    if (!token) {
      throw { statusCode: 401, message: 'Authentication required' };
    }

    const decoded = verifyToken(token);

    // Attach user info to the request object
    (req as any).user = decoded;

    next();
  } catch (error) {
    console.error('Authentication error:', error);
    res.status(401).json({ error: 'Invalid token' });
  }
};
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/pages/api/upload_file.ts

## Summary

This JavaScript file is responsible for handling file analysis using a form recognizer. It defines functions that process file uploads and return analysis results, integrating with a web API to facilitate document analysis in a structured manner.

## Changes Made

- Added function to analyze files with a form recognizer
- Implemented handler function for API requests

```typescript
import type { NextApiRequest, NextApiResponse } from 'next';
import formidable from 'formidable';
import fs from 'fs';
import { authenticate } from '@/utils/auth';
import { AzureKeyCredential, DocumentAnalysisClient } from '@azure/ai-form-recognizer';
import clientPromise from '@/utils/mongodb';
import Papa from 'papaparse';
import { analyzeFileContent } from '@/utils/azure';
import { apiHandler } from '@/utils/apiHandler';
import { AnalysisResult, ErrorResponse } from '../../../frontend/src/types';

export const config = {
  api: {
    bodyParser: false,
  },
};

const analyzeFileWithFormRecognizer = async (file: formidable.File): Promise<AnalysisResult> => {
  const endpoint = process.env.FORM_RECOGNIZER_ENDPOINT || '';
  const apiKey = process.env.FORM_RECOGNIZER_API_KEY || '';

  if (!endpoint || !apiKey) {
    throw { statusCode: 500, message: 'Form Recognizer configuration is missing.' };
  }

  const credential = new AzureKeyCredential(apiKey);
  const client = new DocumentAnalysisClient(endpoint, credential);

  const fileBuffer = fs.readFileSync(file.filepath);
  const fileBlob = new Blob([fileBuffer], { type: file.type });

  try {
    const fileExtension = file.name.split('.').pop()?.toLowerCase();

    if (fileExtension === 'csv') {
      const csvText = await fileBlob.text();
      const parsedData = Papa.parse(csvText, { header: true }).data;
      return { analysis: JSON.stringify({ csvData: parsedData }) };
    } else if (['txt', 'md', 'js', 'py'].includes(fileExtension || '')) {
      const analysis = await analyzeFileContent(file.filepath);
      return { analysis };
    } else {
      const poller = await client.beginAnalyzeDocument('prebuilt-document', fileBlob);
      const result = await poller.pollUntilDone();

      if (!result) {
        throw { statusCode: 500, message: 'Failed to get analysis results from Form Recognizer.' };
      }

      return {
        analysis: JSON.stringify({
          keyValuePairResults: result.keyValuePairResults,
          tables: result.tables,
        })
      };
    }
  } catch (error: any) {
    console.error('Error analyzing file:', error);
    throw { statusCode: 500, message: `Failed to analyze file: ${error.message}` };
  }
};

const handler = async (req: NextApiRequest, res: NextApiResponse<AnalysisResult | ErrorResponse>) => {
  const user = authenticate(req, res);
  if (!user) {
    throw { statusCode: 401, message: 'Unauthorized: Invalid or missing authentication token.' };
  }

  if (req.method !== 'POST') {
    throw { statusCode: 405, message: `Method ${req.method} Not Allowed. Use POST to upload files.` };
  }

  const form = new formidable.IncomingForm();
  form.maxFileSize = parseFloat(process.env.MAX_FILE_SIZE_MB || '5') * 1024 * 1024;
  form.keepExtensions = true;

  return new Promise((resolve, reject) => {
    form.parse(req, async (err, fields, files) => {
      if (err) {
        console.error('Formidable Error:', err);
        reject({ statusCode: 400, message: `Error parsing the uploaded file: ${err.message}` });
        return;
      }

      const file = files.file as formidable.File;
      if (!file) {
        reject({ statusCode: 400, message: 'No file uploaded. Please select a file to upload.' });
        return;
      }

      const allowedExtensions = (process.env.ALLOWED_EXTENSIONS || 'txt,csv,docx,md,js,py,pdf').split(',').map(ext => ext.trim().toLowerCase());
      const fileExtension = file.name.split('.').pop()?.toLowerCase();

      if (!fileExtension || !allowedExtensions.includes(fileExtension)) {
        reject({ statusCode: 400, message: `File type not allowed. Allowed types are: ${allowedExtensions.join(', ')}. Uploaded file type: ${fileExtension || 'unknown'}` });
        return;
      }

      try {
        const analysis = await analyzeFileWithFormRecognizer(file);

        const client = await clientPromise;
        const db = client.db(process.env.MONGODB_DB_NAME);
        const uploads = db.collection('uploads');

        await uploads.insertOne({
          user_id: user.id,
          filename: file.name,
          filepath: file.filepath,
          analysis,
          uploaded_at: new Date(),
        });

        res.status(200).json({ analysis: `File uploaded and analyzed successfully. ${analysis.analysis}` });
        resolve();
      } catch (error: any) {
        console.error('Upload File Error:', error);
        reject({ statusCode: 500, message: `Failed to upload and analyze the file: ${error.message}` });
      }
    });
  });
};

export default apiHandler(handler);
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/middleware/errorHandler.ts

## Summary

This JavaScript file contains utility functions for handling errors in asynchronous operations. The primary function, `errorHandler`, is designed to wrap asynchronous functions and catch any errors that may occur during their execution, allowing for centralized error management and logging within the application.

## Changes Made

- Added errorHandler function for improved error handling in asynchronous operations.

```typescript
// File: apps/backend/middleware/errorHandler.ts

import type { NextApiRequest, NextApiResponse } from 'next';

interface CustomError extends Error {
  status?: number;
}

/**
 * Centralized error handling middleware for API routes.
 * Catches errors thrown in the route handler and formats the response.
 * @param fn - The API route handler function.
 * @returns A wrapped API route handler with error handling.
 */
export const errorHandler = (fn: Function) => {
  return async (req: NextApiRequest, res: NextApiResponse) => {
    try {
      await fn(req, res);
    } catch (error: any) {
      const customError: CustomError = error;
      const status = customError.status || 500;
      const message = customError.message || 'Internal Server Error';

      res.status(status).json({ message });
    }
  };
};
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/pages/api/protected_route.ts

## Summary

This JavaScript file defines an API handler function for processing requests and sending responses in a Next.js application. It serves as a middleware to manage incoming requests and provide appropriate responses based on the application's logic.

## Changes Made

- Added handler function to manage API requests and responses.

```typescript
import { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/middleware/auth';
import { apiHandler } from '@/utils/apiHandler';

interface ProtectedRouteResponse {
  message: string;
}

const handler = async (req: NextApiRequest, res: NextApiResponse<ProtectedRouteResponse>) => {
  authenticate(req, res, () => {
    res.status(200).json({ message: 'You have access to this protected route' });
  });
};

export default apiHandler(handler);
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/pages/api/reset_conversation.ts

## Summary

This JavaScript file contains an API handler function designed to process incoming requests and send appropriate responses. It is part of a larger application that utilizes Next.js for server-side rendering and API routes.

## Changes Made

- Added handler function to manage API requests and responses.

```typescript
import { NextApiRequest, NextApiResponse } from 'next';
import { connectToDatabase } from '../../utils/mongodb';
import { ObjectId } from 'mongodb';
import { apiHandler } from '../../utils/apiHandler';

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  if (req.method !== 'POST') {
    throw { statusCode: 405, message: 'Method not allowed' };
  }

  const { conversation_id } = req.body;

  if (!conversation_id) {
    throw { statusCode: 400, message: 'Conversation ID is required' };
  }

  const { db } = await connectToDatabase();
  const conversationsCollection = db.collection('conversations');

  // Find the conversation and reset its messages
  const result = await conversationsCollection.updateOne(
    { _id: new ObjectId(conversation_id) },
    { $set: { messages: [] } }
  );

  if (result.modifiedCount === 0) {
    throw { statusCode: 404, message: 'Conversation not found' };
  }

  res.status(200).json({ message: 'Conversation reset successfully' });
};

export default apiHandler(handler);
```

# File: /mnt/disks/data/chatapp-vercel/apps/backend/pages/api/list_conversations.ts

## Summary

This JavaScript file defines an API handler function for managing conversations in a Next.js application. It processes incoming requests and sends appropriate responses based on the application's logic.

## Changes Made

- Added handler function to manage API requests for conversations.

```typescript
// File: apps/backend/pages/api/list_conversations.ts

import type { NextApiRequest, NextApiResponse } from 'next';
import { authenticate } from '@/utils/auth';
import clientPromise from '@/utils/mongodb';
import { Conversation } from '@/types/models';
import { apiHandler } from '@/utils/apiHandler';

interface ListConversationsResponse {
  conversations: Conversation[];
}

const handler = async (req: NextApiRequest, res: NextApiResponse<ListConversationsResponse>) => {
  const user = authenticate(req, res);
  if (!user) {
    throw { statusCode: 401, message: 'Unauthorized' };
  }

  if (req.method !== 'GET') {
    throw { statusCode: 405, message: `Method ${req.method} Not Allowed` };
  }

  const client = await clientPromise;
  const db = client.db(process.env.MONGODB_DB_NAME);
  const conversations = db.collection('conversations');

  const userConversations = await conversations
    .find({ user_id: user.id })
    .sort({ updated_at: -1 })
    .toArray();

  res.status(200).json({ conversations: userConversations });
};

export default apiHandler(handler);
```

