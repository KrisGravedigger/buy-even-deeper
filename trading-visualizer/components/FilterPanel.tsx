import React from 'react';
import { type EnhancedStrategyGroup, type SortField } from '../types';

export interface NumericFilter {
  type: 'greater' | 'less' | 'equal';
  value: number;
}

export interface TextFilter {
  type: 'contains' | 'not_contains';
  value: string;
}

export interface ValueListFilter {
  type: 'value_list';
  values: Set<string>;
}

export type ColumnFilter = NumericFilter | TextFilter | ValueListFilter;

export interface ColumnDefinition {
  id: SortField;
  header: string;
  isNumeric?: boolean;
  filter: ColumnFilter | null;
  frozen: boolean;
  getValue: (group: EnhancedStrategyGroup) => number | string | boolean;
  formatDisplay?: (value: number) => string;
}

interface FilterPanelProps {
  columns: ColumnDefinition[];
  onFilterChange: (columnId: string, filter: ColumnFilter | null) => void;
}

export const FilterPanel: React.FC<FilterPanelProps> = ({ columns, onFilterChange }) => {
  const handleFilterTypeChange = (columnId: string, filterType: string) => {
    const column = columns.find(col => col.id === columnId);
    if (!column) return;

    if (filterType === 'none') {
      onFilterChange(columnId, null);
      return;
    }

    if (column.isNumeric) {
      const currentFilter = column.filter as NumericFilter;
      const newFilter: NumericFilter = {
        type: filterType as 'greater' | 'less' | 'equal',
        value: currentFilter?.value || 0
      };
      onFilterChange(columnId, newFilter);
    } else {
      const currentFilter = column.filter as TextFilter;
      const newFilter: TextFilter = {
        type: filterType as 'contains' | 'not_contains',
        value: currentFilter?.value || ''
      };
      onFilterChange(columnId, newFilter);
    }
  };

  const handleFilterValueChange = (columnId: string, value: string) => {
    const column = columns.find(col => col.id === columnId);
    if (!column || !column.filter) return;

    if (column.filter.type === 'value_list') {
      // Obsługa ValueListFilter
      const newFilter: ValueListFilter = {
        type: 'value_list',
        values: new Set([value]) // lub inna logika dla listy wartości
      };
      onFilterChange(columnId, newFilter);
    } else if (column.isNumeric) {
      const newFilter: NumericFilter = {
        ...column.filter as NumericFilter,
        value: Number(value)
      };
      onFilterChange(columnId, newFilter);
    } else {
      const newFilter: TextFilter = {
        ...column.filter as TextFilter,
        value: value
      };
      onFilterChange(columnId, newFilter);
    }
  };

  return (
    <div className="p-4 space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {columns.map(column => (
          <div key={column.id} className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-700 min-w-[120px]">
              {column.header}:
            </span>
            <select
              className="rounded-md border-gray-300 shadow-sm text-sm"
              value={column.filter?.type || 'none'}
              onChange={(e) => handleFilterTypeChange(column.id, e.target.value)}
            >
              <option value="none">Brak filtru</option>
              {column.isNumeric ? (
                <>
                  <option value="greater">Większe niż</option>
                  <option value="less">Mniejsze niż</option>
                  <option value="equal">Równe</option>
                </>
              ) : (
                <>
                  <option value="contains">Zawiera</option>
                  <option value="not_contains">Nie zawiera</option>
                </>
              )}
            </select>
            {column.filter && column.filter.type !== 'value_list' && (
              <input
                type={column.isNumeric ? 'number' : 'text'}
                className="rounded-md border-gray-300 shadow-sm text-sm w-48"
                value={
                  (column.filter as NumericFilter | TextFilter).value
                }
                onChange={(e) => handleFilterValueChange(column.id, e.target.value)}
                placeholder={`Wartość dla ${column.header}`}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
