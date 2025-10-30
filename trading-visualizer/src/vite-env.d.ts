/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_APP_TITLE: string
  // więcej zmiennych środowiskowych...
}

// Rozszerzamy istniejący interfejs ImportMeta tylko o brakujące właściwości
declare module 'vite/client' {
  interface ImportMeta {
    readonly globSync: <T>(pattern: string) => Record<string, T>;
  }
} 