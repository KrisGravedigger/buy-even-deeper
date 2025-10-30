export interface BTCMetrics {
  btc_block_effectiveness: number;
  btc_correlation: number;
  btc_block_rate: number;
}

export interface Metrics {
  trade_count: number;
  win_rate: number;
  avg_profit: number;
  total_profit: number;
  max_drawdown: number;
  btc_metrics: BTCMetrics;
}

export interface BaseStrategy {
  strategy_id: string;
  symbol: string;
  metrics: Metrics;
  score: number;
}

export interface ParameterRange {
  values: (number | boolean)[];
  range: [number, number];
}

export interface FileMetadata {
  fileName: string;
  timestamp: string;
  analysisId: string;
}

export interface EnhancedStrategyGroup {
  group_id: string;
  similar_count: number;
  base_strategy: BaseStrategy;
  parameters: {
    static: Record<string, number | boolean>;
    variable: Record<string, ParameterRange>;
  };
  fileMetadata: FileMetadata;
}

export type SortField = keyof Metrics | 'strategy_id' | 'symbol' | 'group_id' | 'similar_count' | 'score' | 'fileName' | 'timestamp' | string;
export type SortDirection = 'asc' | 'desc';

export interface SymbolSummary {
  total_strategies: number;
  total_groups: number;
  avg_win_rate: number;
  avg_profit: number;
  best_score: number;
  worst_score: number;
}

export interface AggregatedSymbolSummary {
  totalStrategies: number;
  totalGroups: number;
  avgWinRate: number;
  avgProfit: number;
  bestScore: number;
  worstScore: number;
}

export interface StrategyGroup {
  group_id: number;
  similar_count: number;
  base_strategy: BaseStrategy;
  parameters: {
    static: Record<string, number | boolean>;
    variable: Record<string, ParameterRange>;
  };
}

export interface TradingData {
  timestamp: string;
  strategy_groups: StrategyGroup[];
  symbols_summary: Record<string, SymbolSummary>;
}

export interface AggregatedTradingData {
  allGroups: EnhancedStrategyGroup[];
  symbolsSummary: Record<string, AggregatedSymbolSummary>;
  filesProcessed: FileMetadata[];
}
