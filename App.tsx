import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { Dashboard } from './components/Dashboard';
import { CompanyAnalysis } from './components/CompanyAnalysis';
import { Valuation } from './components/Valuation';
import { LoadingSpinner } from './components/LoadingSpinner';
import type { NavItem, CompanyData } from './types';
import { NAV_ITEMS } from './constants';
import { getCompanyAnalysis } from './services/geminiService';

function App() {
  const [activeNav, setActiveNav] = useState<NavItem>(NAV_ITEMS[0]);
  const [isSidebarOpen, setSidebarOpen] = useState(false);
  const [companyData, setCompanyData] = useState<CompanyData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (companyName: string, isIndian: boolean) => {
    setIsLoading(true);
    setError(null);
    setCompanyData(null);
    setActiveNav(NAV_ITEMS[0]); // Switch to analysis view
    try {
      const data = await getCompanyAnalysis(companyName, isIndian);
      setCompanyData(data);
    } catch (e) {
      if (e instanceof Error) {
        setError(e.message);
      } else {
        setError(`An unknown error occurred while analyzing ${companyName}.`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex flex-col items-center justify-center h-full">
          <LoadingSpinner />
          <p className="mt-4 text-slate-300">Analyzing company data...</p>
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="bg-red-900/50 text-red-300 p-6 rounded-lg text-center max-w-lg">
            <h2 className="text-xl font-bold mb-2">An Error Occurred</h2>
            <p>{error}</p>
          </div>
        </div>
      );
    }
    
    if (!companyData) {
      return <Dashboard onAnalyze={handleAnalyze} />;
    }

    switch (activeNav.id) {
      case 'analysis':
      case 'news':
      case 'reports': // For now, reports link to analysis
        return <CompanyAnalysis data={companyData} activeNavId={activeNav.id} />;
      case 'valuation':
        return <Valuation companyData={companyData} />;
      default:
        return <Dashboard onAnalyze={handleAnalyze} />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex font-sans">
      <Sidebar 
        activeNav={activeNav} 
        setActiveNav={setActiveNav} 
        isOpen={isSidebarOpen} 
        setIsOpen={setSidebarOpen} 
      />
      <div className="flex-1 flex flex-col">
        <Header onAnalyze={handleAnalyze} onToggleSidebar={() => setSidebarOpen(!isSidebarOpen)} />
        <main className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8">
          {renderContent()}
        </main>
      </div>
    </div>
  );
}

export default App;