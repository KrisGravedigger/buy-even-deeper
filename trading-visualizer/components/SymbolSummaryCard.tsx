import React from 'react';
import { type AggregatedSymbolSummary } from '../types';

interface SymbolSummaryCardProps {
  symbol: string;
  summary: AggregatedSymbolSummary;
}

export const SymbolSummaryCard: React.FC<SymbolSummaryCardProps> = ({ symbol, summary }) => {
  return (
    <div className="bg-white overflow-hidden shadow rounded-lg p-6 w-full max-w-[1400px]">
      <dl className="flex flex-row items-center justify-between gap-8">
        <div className="min-w-[100px]">
          <dd className="text-lg font-semibold text-gray-900">{symbol}</dd>
        </div>
        <div className="min-w-[120px]">
          <dt className="text-sm font-medium text-gray-500 whitespace-nowrap">Liczba strategii</dt>
          <dd className="mt-1 text-lg font-semibold text-gray-900">{summary.totalStrategies}</dd>
        </div>
        <div className="min-w-[120px]">
          <dt className="text-sm font-medium text-gray-500 whitespace-nowrap">Liczba grup</dt>
          <dd className="mt-1 text-lg font-semibold text-gray-900">{summary.totalGroups}</dd>
        </div>
        <div className="min-w-[120px]">
          <dt className="text-sm font-medium text-gray-500 whitespace-nowrap">Średni Win Rate</dt>
          <dd className="mt-1 text-lg font-semibold text-gray-900">{summary.avgWinRate.toFixed(2)}%</dd>
        </div>
        <div className="min-w-[120px]">
          <dt className="text-sm font-medium text-gray-500 whitespace-nowrap">Średni zysk</dt>
          <dd className="mt-1 text-lg font-semibold text-gray-900">{summary.avgProfit.toFixed(2)}%</dd>
        </div>
      </dl>
    </div>
  );
};