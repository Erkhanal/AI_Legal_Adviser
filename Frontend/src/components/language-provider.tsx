'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import enTranslations from '@/locales/en.json';
import neTranslations from '@/locales/ne.json';
import { enUS } from 'date-fns/locale/en-US';
import { type Locale } from 'date-fns';

type Language = 'en' | 'ne';

const translations: Record<Language, Record<string, string>> = {
  en: enTranslations,
  ne: neTranslations,
};

const dateLocales: Record<Language, Locale> = {
    en: enUS,
    ne: enUS, // Fallback to enUS to resolve build error
}

interface LanguageContextType {
  language: Language;
  setLanguage: (language: Language) => void;
  t: (key: string, replacements?: { [key: string]: string | number }) => string;
  dateLocale: Locale;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>('en');

  useEffect(() => {
    const storedLang = localStorage.getItem('language') as Language | null;
    if (storedLang && ['en', 'ne'].includes(storedLang)) {
      setLanguage(storedLang);
    }
  }, []);

  const handleSetLanguage = (lang: Language) => {
    setLanguage(lang);
    localStorage.setItem('language', lang);
  };

  const t = (key: string, replacements?: { [key: string]: string | number }) => {
    let translation = translations[language]?.[key] || translations['en']?.[key] || key;
    
    if (replacements) {
        Object.keys(replacements).forEach(rKey => {
            const regex = new RegExp(`{${rKey}}`, 'g');
            translation = translation.replace(regex, String(replacements[rKey]));
        });
    }
    return translation;
  };
  
  const dateLocale = dateLocales[language];

  return (
    <LanguageContext.Provider value={{ language, setLanguage: handleSetLanguage, t, dateLocale }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useTranslation() {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useTranslation must be used within a LanguageProvider');
  }
  return context;
}
