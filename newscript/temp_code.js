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