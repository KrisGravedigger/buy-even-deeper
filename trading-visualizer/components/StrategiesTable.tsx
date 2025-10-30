import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import TableHeader from './TableHeader';
import { type ColumnDefinition, type ColumnFilter } from './FilterPanel';
import { type EnhancedStrategyGroup, type SortField, type SortDirection } from '../types';

// Utility functions
const normalizeBoolean = (value: any): number => {
  if (value === undefined || value === null) return 0;
  if (typeof value === 'boolean') return value ? 1 : 0;
  if (value === 0.0 || value === 1.0) return Number(value);
  return 0;
};

const formatBooleanDisplay = (value: number): string => {
  return value === 1 ? 'Tak' : 'Nie';
};

interface StrategiesTableProps {
  groups: EnhancedStrategyGroup[];
}

interface ColumnStatistics {
  values: number[];
  percentile70: number;
  percentile20: number;
  frequencyMap: Map<number, number>;
  totalCount: number;
  isInverse: boolean;
}



export default function StrategiesTable({ groups }: StrategiesTableProps) {
  const [sortField, setSortField] = useState<SortField>('score');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const [showAllDetails, setShowAllDetails] = useState(false);
  const [hiddenRows, setHiddenRows] = useState<Set<string>>(new Set());
  const [columns, setColumns] = useState<ColumnDefinition[]>(() => [
    // Zachowujemy istniejące definicje kolumn
    {
      id: 'fileName',
      header: 'Plik źródłowy',
      frozen: false,
      filter: null,
      getValue: (group) => group.fileMetadata.fileName,
      isNumeric: false
    },
    {
      id: 'timestamp',
      header: 'Data analizy',
      frozen: false,
      filter: null,
      getValue: (group) => group.fileMetadata.timestamp,
      isNumeric: false
    },
    {
      id: 'group_id',
      header: 'ID Grupy',
      frozen: false,
      filter: null,
      getValue: (group) => group.group_id,
      isNumeric: false
    },
    {
      id: 'similar_count',
      header: 'Podobne strategie',
      frozen: false,
      filter: null,
      getValue: (group) => group.similar_count,
      isNumeric: true
    },
    {
      id: 'strategy_id',
      header: 'ID Strategii',
      frozen: false,
      filter: null,
      getValue: (group) => group.base_strategy.strategy_id,
      isNumeric: false
    },
    {
      id: 'symbol',
      header: 'Symbol',
      frozen: false,
      filter: null,
      getValue: (group) => group.base_strategy.symbol,
      isNumeric: false
    },
    {
      id: 'score',
      header: 'Score',
      frozen: false,
      filter: null,
      getValue: (group) => group.base_strategy.score,
      isNumeric: true
    },
    {
      id: 'trade_count',
      header: 'Liczba transakcji',
      frozen: false,
      filter: null,
      getValue: (group) => group.base_strategy.metrics.trade_count,
      isNumeric: true
    },
    {
      id: 'win_rate',
      header: 'Win Rate (%)',
      frozen: false,
      filter: null,
      getValue: (group) => group.base_strategy.metrics.win_rate,
      isNumeric: true
    },
    {
      id: 'avg_profit',
      header: 'Średni zysk (%)',
      frozen: false,
      filter: null,
      getValue: (group) => group.base_strategy.metrics.avg_profit,
      isNumeric: true
    },
    {
      id: 'total_profit',
      header: 'Całkowity zysk (%)',
      frozen: false,
      filter: null,
      getValue: (group) => group.base_strategy.metrics.total_profit,
      isNumeric: true
    },
    {
      id: 'max_drawdown',
      header: 'Max Drawdown (%)',
      frozen: false,
      filter: null,
      getValue: (group) => group.base_strategy.metrics.max_drawdown,
      isNumeric: true
    },
    {
      id: 'trailing_buy_enabled',
      header: 'Trailing Buy',
      frozen: false,
      filter: null,
      getValue: (group) => normalizeBoolean(group.parameters.static.trailing_buy_enabled),
      isNumeric: true,
      formatDisplay: formatBooleanDisplay
    },
    {
      id: 'trailing_buy_threshold',
      header: 'Trailing Buy Threshold',
      frozen: false,
      filter: null,
      getValue: (group) => Number(group.parameters.static.trailing_buy_threshold ?? 0),
      isNumeric: true
    },
    {
      id: 'trailing_buy_time_in_min',
      header: 'Trailing Buy Time',
      frozen: false,
      filter: null,
      getValue: (group) => Number(group.parameters.static.trailing_buy_time_in_min ?? 0),
      isNumeric: true
    }
  ]);

  const [paramColumns, setParamColumns] = useState<ColumnDefinition[]>(() => []);
  
  const tableContainerRef = useRef<HTMLDivElement>(null);

  const tableHeight = 'calc(100vh - 200px)';

  // Funkcja pomocnicza do sprawdzania czy wiersz powinien być rozwinięty
  const shouldShowDetails = useCallback((groupId: string) => {
    return showAllDetails || expandedRows.has(groupId);
  }, [showAllDetails, expandedRows]);

  // Aktualizujemy useEffect dla parameterColumns
  useEffect(() => {
    if (!groups.length) return;
    
    const firstGroup = groups[0];
    const staticParams = Object.keys(firstGroup.parameters.static);
    const variableParams = Object.keys(firstGroup.parameters.variable);
    
    const newParamColumns = [
      ...staticParams.map(param => ({
        id: `static_${param}` as SortField,
        header: param,
        getValue: (group: EnhancedStrategyGroup) => {
          const value = group.parameters.static[param];
          // Obsługa wartości boolean
          if (typeof value === 'boolean') {
            return value ? 1 : 0;
          }
          return value;
        },
        filter: null,
        frozen: false,
        isNumeric: true
      })),
      ...variableParams.map(param => ({
        id: `variable_${param}` as SortField,
        header: `${param} (range)`,
        getValue: (group: EnhancedStrategyGroup) => {
          const paramData = group.parameters.variable[param];
          // Sprawdzamy czy parametr ma wartości i zakres
          if (paramData?.range && paramData.range.length >= 2) {
            return paramData.range[0];
          }
          // Jeśli nie ma zakresu, zwracamy pierwszą wartość z values
          if (paramData?.values && paramData.values.length > 0) {
            return paramData.values[0];
          }
          return 0;
        },
        filter: null,
        frozen: false,
        isNumeric: true
      }))
    ] as ColumnDefinition[];
  
    setParamColumns(newParamColumns);
  }, [groups]);

  useEffect(() => {
    console.log('paramColumns:', paramColumns);
    console.log('expandedRows:', Array.from(expandedRows));
    console.log('showAllDetails:', showAllDetails);
  }, [paramColumns, expandedRows, showAllDetails]);

  

  // Funkcja do zapamiętywania pozycji scrolla
  const preserveScroll = (callback: () => void) => {
    if (!tableContainerRef.current) return callback();

    const container = tableContainerRef.current;
    const scrollLeft = container.scrollLeft;
    const scrollTop = container.scrollTop;

    callback();

    // Przywracamy pozycję scrolla w następnym cyklu renderowania
    requestAnimationFrame(() => {
      container.scrollLeft = scrollLeft;
      container.scrollTop = scrollTop;
    });
  };

  // Obliczanie statystyk dla kolumn numerycznych
  const columnStatistics = useMemo(() => {
    const stats = new Map<string, ColumnStatistics>();
    
    columns.forEach(column => {
      if (column.isNumeric) {
        const values = groups.map(group => column.getValue(group) as number);
        const isInverse = column.id === 'max_drawdown';
        const sortedValues = [...values].sort((a, b) => 
          isInverse ? a - b : b - a
        );
        
        // Obliczanie percentyli
        const index70 = Math.floor(sortedValues.length * 0.3); // top 30%
        const index20 = Math.floor(sortedValues.length * 0.8); // bottom 20%
        const percentile70 = sortedValues[index70];
        const percentile20 = sortedValues[index20];
        
        // Obliczanie częstości występowania wartości
        const frequencyMap = new Map<number, number>();
        values.forEach(value => {
          frequencyMap.set(value, (frequencyMap.get(value) || 0) + 1);
        });
        
        stats.set(column.id, {
          values: sortedValues,
          percentile70,
          percentile20,
          frequencyMap,
          totalCount: values.length,
          isInverse
        });
      }
    });
    
    return stats;
  }, [columns, groups]);

  // Pomocnicza funkcja do sprawdzania filtrów
  const checkFilter = (column: ColumnDefinition, group: EnhancedStrategyGroup): boolean => {
    const value = column.getValue(group);
    const filter = column.filter;
    
    if (!filter) return true;

    if (filter.type === 'value_list') {
      return filter.values.has(String(value));
    }

    if (column.isNumeric && (typeof value === 'number' || typeof value === 'boolean')) {
      const numericValue = typeof value === 'boolean' ? (value ? 1 : 0) : value;
      const filterValue = typeof filter.value === 'string' ? parseFloat(filter.value) : filter.value;
      
      if (isNaN(filterValue)) return true;

      switch (filter.type) {
        case 'greater':
          return numericValue > filterValue;
        case 'less':
          return numericValue < filterValue;
        case 'equal':
          return numericValue === filterValue;
        default:
          return true;
      }
    } else {
      const stringValue = String(value).toLowerCase();
      const filterValue = String(filter.value).toLowerCase();
      
      switch (filter.type) {
        case 'contains':
          return stringValue.includes(filterValue);
        case 'not_contains':
          return !stringValue.includes(filterValue);
        default:
          return true;
      }
    }
  };

  // Filtrowanie danych
  const filteredData = useMemo(() => {
    return groups.filter(group => {
      // Sprawdzamy filtry głównych kolumn
      const mainColumnsMatch = columns.every(column => {
        if (!column.filter) return true;
        return checkFilter(column, group);
      });
    
      // Sprawdzamy filtry kolumn parametrów tylko jeśli są rozwinięte
      const paramColumnsMatch = expandedRows.size === 0 || paramColumns.every(column => {
        if (!column.filter) return true;
        return checkFilter(column, group);
      });
    
      return mainColumnsMatch && paramColumnsMatch;
    });
  }, [groups, columns, paramColumns, expandedRows]);

  // Sortowanie przefiltrowanych danych
  const sortedData = useMemo(() => {
    const allColumns = [...columns, ...(expandedRows.size > 0 ? paramColumns : [])];
    const column = allColumns.find(col => col.id === sortField);
    if (!column) return filteredData;

    // Rozdzielamy dane na rozwinięte i nierozwinięte
    const expanded = filteredData.filter(group => shouldShowDetails(group.group_id));
    const collapsed = filteredData.filter(group => !shouldShowDetails(group.group_id));

    // Funkcja pomocnicza do sortowania
    const sortGroups = (groups: EnhancedStrategyGroup[]) => {
      return [...groups].sort((a, b) => {
        const aValue = column.getValue(a);
        const bValue = column.getValue(b);

        if (typeof aValue === 'string' && typeof bValue === 'string') {
          return sortDirection === 'asc' 
            ? aValue.localeCompare(bValue)
            : bValue.localeCompare(aValue);
        }

        return sortDirection === 'asc' 
          ? (aValue as number) - (bValue as number)
          : (bValue as number) - (aValue as number);
      });
    };

    // Sortujemy obie grupy osobno i łączymy je
    const sortedExpanded = sortGroups(expanded);
    const sortedCollapsed = sortGroups(collapsed);

    return [...sortedExpanded, ...sortedCollapsed];
  }, [filteredData, sortField, sortDirection, columns, expandedRows, paramColumns, shouldShowDetails]);

  // Funkcja do sprawdzania unikalności wartości parametru
  const isParameterValueUnique = (columnId: string, value: number | string | boolean, visibleGroups: EnhancedStrategyGroup[]) => {
    // Convert boolean to number if needed
    const compareValue = typeof value === 'boolean' ? (value ? 1 : 0) : value;
    if (typeof compareValue !== 'number' && typeof compareValue !== 'string') {
      return false;
    }

    const expandedAndVisibleGroups = visibleGroups.filter(group => 
      shouldShowDetails(group.group_id) && !hiddenRows.has(group.group_id)
    );
    
    const matchingGroups = expandedAndVisibleGroups.filter(group => {
      const column = paramColumns.find(col => col.id === columnId);
      if (!column) return false;

      let columnValue = column.getValue(group);
      // Convert boolean values to numbers for comparison
      if (typeof columnValue === 'boolean') {
        columnValue = columnValue ? 1 : 0;
      }
      
      if (typeof columnValue !== 'number' && typeof columnValue !== 'string') {
        return false;
      }

      return columnValue === compareValue;
    });
    
    return matchingGroups.length === 1;
  };

  const handleSort = useCallback((field: SortField) => {
    if (sortField === field) {
      setSortDirection(current => current === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  }, [sortField]);

  const handleToggleFreeze = (columnId: string) => {
    preserveScroll(() => {
      setColumns(prevColumns => {
        // Count currently frozen columns
        const frozenCount = prevColumns.filter(col => col.frozen).length;
        // Get the column we're trying to toggle
        const targetColumn = prevColumns.find(col => col.id === columnId);
        
        if (!targetColumn) return prevColumns;
        
        // If trying to freeze and already have one frozen, unfreeze the currently frozen one
        if (!targetColumn.frozen && frozenCount >= 1) {
          return prevColumns.map(col => ({
            ...col,
            frozen: col.id === columnId
          }));
        }
        
        // Otherwise just toggle the target column
        return prevColumns.map(col => 
          col.id === columnId ? { ...col, frozen: !col.frozen } : col
        );
      });
    });
  };

  const handleFilterChange = (columnId: string, filter: ColumnFilter | null) => {
    preserveScroll(() => {
      const isParameterColumn = paramColumns.some(col => col.id === columnId);
      if (isParameterColumn) {
        setParamColumns(prevColumns =>
          prevColumns.map(col =>
            col.id === columnId ? { ...col, filter } : col
          )
        );
      } else {
        setColumns(prevColumns =>
          prevColumns.map(col =>
            col.id === columnId ? { ...col, filter } : col
          )
        );
      }
    });
  };

  const handleClearAllFilters = () => {
    preserveScroll(() => {
      setColumns(prevColumns =>
        prevColumns.map(col => ({ ...col, filter: null }))
      );
    });
  };

  const getFrozenColumnOffset = (columnIndex: number): number => {
    let offset = 80; // szerokość kolumn szczegóły + ukryj
    for (let i = 0; i < columnIndex; i++) {
      if (columns[i].frozen) {
        offset += 200; // stała szerokość kolumny
      }
    }
    return offset;
  };

  const formatNumber = (num: number) => {
    if (typeof num !== 'number' || isNaN(num)) return '0.00';
    
    // Używamy Intl.NumberFormat z explicit locale
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(num);
  };

  const toggleRowDetails = (groupId: string) => {
    preserveScroll(() => {
      if (showAllDetails) {
        const newExpandedRows = new Set(filteredData.map(group => group.group_id));
        newExpandedRows.delete(groupId);
        setExpandedRows(newExpandedRows);
        setShowAllDetails(false);
      } else {
        const newExpandedRows = new Set(expandedRows);
        if (newExpandedRows.has(groupId)) {
          newExpandedRows.delete(groupId);
        } else {
          newExpandedRows.add(groupId);
        }
        setExpandedRows(newExpandedRows);
      }
    });
  };

  const getCellStyle = (columnId: string, value: number) => {
    const stats = columnStatistics.get(columnId);
    if (!stats) return {};

    const style: React.CSSProperties = {};
    
    // Pogrubienie dla najlepszych wartości (uwzględniając odwrotną logikę dla max_drawdown)
    if (stats.isInverse ? value <= stats.percentile20 : value >= stats.percentile70) {
      style.fontWeight = 'bold';
    }

    // Kolorowanie górnych 30% na czerwono i dolnych 20% na niebiesko
    if (stats.isInverse ? value >= stats.percentile70 : value >= stats.percentile70) {
      style.color = 'rgb(239 68 68)'; // text-red-500
    } else if (stats.isInverse ? value <= stats.percentile20 : value <= stats.percentile20) {
      style.color = 'rgb(59 130 246)'; // text-blue-500
    }

    return style;
  };

  return (
    <div>
      {/* Panel akcji */}
      <div className="flex items-center space-x-4 mb-4 sticky top-0 left-0 z-40 bg-white p-4 border-b">
        <button
          onClick={() => {
            preserveScroll(() => {
              setShowAllDetails(!showAllDetails);
              if (!showAllDetails) {
                setExpandedRows(new Set(filteredData.map(group => group.group_id)));
              } else {
                setExpandedRows(new Set());
              }
            });
          }}
          className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
        >
          {showAllDetails ? 'Zwiń wszystkie' : 'Rozwiń wszystkie'}
        </button>
        
        <button
          onClick={handleClearAllFilters}
          className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 flex items-center space-x-2"
        >
          <span>Wyczyść wszystkie filtry</span>
        </button>

        <button
          onClick={() => setHiddenRows(new Set())}
          className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
        >
          Przywróć ukryte
        </button>
      </div>

      {/* Tabela */}
      <div 
        ref={tableContainerRef}
        className="relative overflow-x-auto"
        style={{ height: tableHeight }}
      >
  <div className="overflow-x-auto overflow-y-auto h-full">
        <table className="w-full divide-y divide-gray-200"> 
          <TableHeader
            columns={[...columns, ...(expandedRows.size > 0 ? paramColumns : [])]}
            sortField={sortField}
            sortDirection={sortDirection}
            onSort={handleSort}
            onToggleFreeze={handleToggleFreeze}
            onFilterChange={handleFilterChange}
            frozenColumnOffset={getFrozenColumnOffset}
            filteredCount={filteredData.length}
            totalCount={groups.length}
            visibleGroups={filteredData}
          />
          
          <tbody className="bg-white divide-y divide-gray-200">
            {sortedData.filter(group => !hiddenRows.has(group.group_id)).map((group) => (
              <tr key={group.group_id} className="hover:bg-gray-50">
                {/* Przycisk rozwijania dla wiersza */}
                <td className="px-3 py-4 sticky left-0 z-20 bg-white">
                  <button
                    onClick={() => toggleRowDetails(group.group_id)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    {shouldShowDetails(group.group_id) ? '−' : '+'}
                  </button>
                </td>

                {/* Checkbox ukrywania wiersza */}
                <td className="px-3 py-4 sticky left-[40px] z-20 bg-white">
                  <input
                    type="checkbox"
                    checked={hiddenRows.has(group.group_id)}
                    onChange={(e) => {
                      const newHiddenRows = new Set(hiddenRows);
                      if (e.target.checked) {
                        newHiddenRows.add(group.group_id);
                      } else {
                        newHiddenRows.delete(group.group_id);
                      }
                      setHiddenRows(newHiddenRows);
                    }}
                    className="h-4 w-4 text-blue-600 rounded border-gray-300"
                  />
                </td>

                {columns.map((column, index) => {
                  const value = column.getValue(group);
                  const style = column.isNumeric 
                    ? getCellStyle(column.id, value as number)
                    : {};
                    
                  return (
                    <td
                      key={column.id}
                      className={`
                        relative px-6 py-4 whitespace-nowrap text-sm
                        ${column.isNumeric ? 'text-right min-w-[120px]' : 'text-left min-w-[150px]'}
                        ${column.frozen ? 'sticky z-20 bg-white shadow-[2px_0_5px_-2px_rgba(0,0,0,0.1)] border-r border-gray-200' : ''}
                        ${column.filter ? 'bg-gray-50' : ''}
                      `}
                      style={{
                        ...(column.frozen ? {
                          left: `${getFrozenColumnOffset(index)}px`,
                          backgroundColor: column.filter ? 'rgb(249 250 251)' : 'white'
                        } : {}),
                        ...style
                      }}
                    >
                      {column.formatDisplay 
                        ? column.formatDisplay(value as number)
                        : column.isNumeric 
                          ? formatNumber(value as number) 
                          : value}
                    </td>
                  );
                })}

                {/* Kolumny parametrów - pokazywane warunkowo */}
                {shouldShowDetails(group.group_id) && (
                  <>
                    {paramColumns.map(column => {
                      const value = column.getValue(group);
                      const isUnique = isParameterValueUnique(column.id, value, sortedData);
                      return (
                        <td
                          key={column.id}
                          className="px-6 py-4 whitespace-nowrap text-sm text-gray-900"
                        >
                          <div className="flex items-center space-x-1">
                            <span>
                              {typeof value === 'boolean' 
                                ? (value ? 'Tak' : 'Nie')
                                : typeof value === 'number' 
                                  ? formatNumber(value) 
                                  : value}
                            </span>
                            {isUnique && (
                              <span className="text-red-500 font-bold">!</span>
                            )}
                          </div>
                        </td>
                      );
                    })}
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
    </div>
  );
}