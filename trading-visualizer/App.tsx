/// <reference types="vite/client" />

import { useEffect, useState } from 'react';
import StrategiesTable from './components/StrategiesTable';
import { SymbolSummaryCard } from './components/SymbolSummaryCard';
import { 
  type EnhancedStrategyGroup, 
  type FileMetadata,
  type TradingData,
  type AggregatedTradingData
} from './types';

function App() {
  const [aggregatedData, setAggregatedData] = useState<AggregatedTradingData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const getAllJsonFiles = async () => {
    try {
      const files = import.meta.glob<Record<string, unknown>>('/public/*.json');
      const fileNames = Object.keys(files).map(f => f.split('/').pop() || '');
      return fileNames;
    } catch (error) {
      console.error('Error getting JSON files:', error);
      throw error;
    }
  };

  const processStrategyGroups = (
    data: TradingData, 
    fileName: string
  ): EnhancedStrategyGroup[] => {
    const timestamp = data.timestamp || new Date().toISOString();
    const analysisId = `${fileName.split('_')[1]}_${timestamp}`;
    
    const metadata: FileMetadata = {
      fileName,
      timestamp,
      analysisId
    };

    return data.strategy_groups.map(group => ({
      ...group,
      fileMetadata: metadata,
      group_id: `${metadata.analysisId}_${group.group_id}`
    }));
  };

  useEffect(() => {
    const aggregateData = (allData: { file: string; data: TradingData }[]): AggregatedTradingData => {
      const allGroups: EnhancedStrategyGroup[] = [];
      const filesProcessed: FileMetadata[] = [];
      const symbolsSummary: AggregatedTradingData['symbolsSummary'] = {};
  
      allData.forEach(({ file, data }) => {
        const enhancedGroups = processStrategyGroups(data, file);
        allGroups.push(...enhancedGroups);
        filesProcessed.push(enhancedGroups[0].fileMetadata);
  
        Object.entries(data.symbols_summary).forEach(([symbol, summary]) => {
          if (!symbolsSummary[symbol]) {
            symbolsSummary[symbol] = {
              totalStrategies: 0,
              totalGroups: 0,
              avgWinRate: 0,
              avgProfit: 0,
              bestScore: -Infinity,
              worstScore: Infinity
            };
          }
          
          symbolsSummary[symbol].totalStrategies += summary.total_strategies;
          symbolsSummary[symbol].totalGroups += summary.total_groups;
          symbolsSummary[symbol].bestScore = Math.max(symbolsSummary[symbol].bestScore, summary.best_score);
          symbolsSummary[symbol].worstScore = Math.min(symbolsSummary[symbol].worstScore, summary.worst_score);
        });
      });
  
      Object.keys(symbolsSummary).forEach(symbol => {
        const relevantGroups = allGroups.filter(g => g.base_strategy.symbol === symbol);
        symbolsSummary[symbol].avgWinRate = 
          relevantGroups.reduce((acc, g) => acc + g.base_strategy.metrics.win_rate, 0) / relevantGroups.length;
        symbolsSummary[symbol].avgProfit = 
          relevantGroups.reduce((acc, g) => acc + g.base_strategy.metrics.avg_profit, 0) / relevantGroups.length;
      });
  
      return { allGroups, symbolsSummary, filesProcessed };
    };

    const loadAllFiles = async () => {
      try {
        const fileNames = await getAllJsonFiles();
        const allData = await Promise.all(
          fileNames.map(async (fileName) => {
            const response = await fetch(`/${fileName}`);
            const data = await response.json();
            return { file: fileName, data };
          })
        );
        
        const aggregated = aggregateData(allData);
        setAggregatedData(aggregated);
      } catch (error) {
        console.error('Error loading data:', error);
        setError(error instanceof Error ? error.message : 'Nieznany błąd');
      } finally {
        setIsLoading(false);
      }
    };

    loadAllFiles();
  }, []);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-red-50">
        <div className="text-center">
          <h1 className="text-2xl text-red-600 mb-4">Wystąpił błąd</h1>
          <p className="text-red-500">{error}</p>
        </div>
      </div>
    );
  }

  if (isLoading || !aggregatedData) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <h1 className="text-2xl text-gray-600 mb-4">Ładowanie danych...</h1>
        </div>
      </div>
    );
  }

  return (
  <div className="min-h-screen bg-gray-50 flex flex-col">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full flex-1 flex flex-col py-8">
      <div className="flex-none">
        <h1 className="text-3xl font-bold text-gray-900">Analiza Strategii Tradingowych</h1>
        <div className="mt-4">
          {Object.entries(aggregatedData.symbolsSummary).map(([symbol, summary]) => (
            <SymbolSummaryCard key={symbol} symbol={symbol} summary={summary} />
          ))}
        </div>
      </div>
      
      <div className="flex-1 overflow-hidden bg-white rounded-lg shadow mt-8">
        <StrategiesTable groups={aggregatedData.allGroups} />
      </div>
    </div>
  </div>
  );
}

export default App;
