import React, { useMemo } from 'react';
import { X } from 'lucide-react';
import { ColumnDefinition, ColumnFilter } from './FilterPanel';
import { type SortField } from '../types';
import { type EnhancedStrategyGroup } from '../types';

// Extend ColumnDefinition to ensure id is of type SortField
interface ExtendedColumnDefinition extends Omit<ColumnDefinition, 'id'> {
  id: SortField;
}

interface TableHeaderProps {
  columns: ExtendedColumnDefinition[];
  sortField: SortField;
  sortDirection: 'asc' | 'desc';
  onSort: (field: SortField) => void;
  onToggleFreeze: (columnId: string) => void;
  onFilterChange: (columnId: string, filter: ColumnFilter | null) => void;
  frozenColumnOffset: (index: number) => number;
  filteredCount: number;
  totalCount: number;
  visibleGroups: EnhancedStrategyGroup[];
}

interface FilterDropdownProps {
  column: ExtendedColumnDefinition;
  uniqueValues: (string | number | boolean)[];
  onFilterChange: (columnId: string, filter: ColumnFilter | null) => void;
  onClose: () => void;
}

const FilterDropdown: React.FC<FilterDropdownProps> = ({ column, uniqueValues, onFilterChange, onClose }) => {
  const [filterType, setFilterType] = React.useState<'value_list' | 'condition'>('value_list');
  const [selectedValues, setSelectedValues] = React.useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = React.useState('');
  const [conditionType, setConditionType] = React.useState<string>(column.isNumeric ? 'greater' : 'contains');
  const [conditionValue, setConditionValue] = React.useState<string>('');

  const filteredValues = useMemo(() => {
    return uniqueValues
      .map(v => v.toString())
      .filter(value => 
        value.toLowerCase().includes(searchTerm.toLowerCase())
      )
      .sort((a, b) => {
        // Próba sortowania numerycznego, jeśli to możliwe
        const numA = Number(a);
        const numB = Number(b);
        if (!isNaN(numA) && !isNaN(numB)) {
          return numA - numB;
        }
        // W przeciwnym razie sortowanie alfabetyczne
        return a.localeCompare(b);
      });
  }, [uniqueValues, searchTerm]);

  const handleSelectAll = () => {
    setSelectedValues(new Set(filteredValues));
  };

  const handleClearAll = () => {
    setSelectedValues(new Set());
  };

  const handleApplyFilter = () => {
    if (filterType === 'value_list') {
      if (selectedValues.size === 0) {
        onFilterChange(column.id, null);
      } else {
        onFilterChange(column.id, {
          type: 'value_list',
          values: selectedValues
        });
      }
    } else {
      if (!conditionValue) {
        onFilterChange(column.id, null);
      } else if (column.isNumeric) {
        onFilterChange(column.id, {
          type: conditionType as 'greater' | 'less' | 'equal',
          value: Number(conditionValue)
        });
      } else {
        onFilterChange(column.id, {
          type: conditionType as 'contains' | 'not_contains',
          value: conditionValue
        });
      }
    }
    onClose();
  };

  return (
    <div className="absolute z-50 mt-1 bg-white border rounded-lg shadow-lg w-64">
      <div className="p-2 border-b flex justify-between items-center">
        <select
          className="w-full px-2 py-1 border rounded text-sm mb-2"
          value={filterType}
          onChange={(e) => setFilterType(e.target.value as 'value_list' | 'condition')}
        >
          <option value="value_list">Lista wartości</option>
          <option value="condition">Warunek</option>
        </select>
        <button
          onClick={onClose}
          className="ml-2 p-1 hover:bg-gray-100 rounded"
          aria-label="Zamknij"
        >
          <X size={16} />
        </button>
      </div>

      {filterType === 'value_list' ? (
        <>
          <input
            type="text"
            className="w-full px-2 py-1 border rounded text-sm"
            placeholder="Szukaj..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <div className="p-2 border-b">
            <button className="text-xs text-blue-600 mr-2" onClick={handleSelectAll}>
              Zaznacz wszystkie
            </button>
            <button className="text-xs text-blue-600" onClick={handleClearAll}>
              Wyczyść wszystkie
            </button>
          </div>
          <div className="max-h-60 overflow-y-auto">
            {filteredValues.map((value) => (
              <label key={value} className="flex items-center px-2 py-1 hover:bg-gray-50 cursor-pointer">
                <input
                  type="checkbox"
                  className="mr-2"
                  checked={selectedValues.has(value)}
                  onChange={(e) => {
                    const newSelected = new Set(selectedValues);
                    if (e.target.checked) {
                      newSelected.add(value);
                    } else {
                      newSelected.delete(value);
                    }
                    setSelectedValues(newSelected);
                  }}
                />
                <span className="text-sm">{value}</span>
              </label>
            ))}
          </div>
        </>
      ) : (
        <div className="space-y-2">
          <select
            className="w-full px-2 py-1 border rounded text-sm"
            value={conditionType}
            onChange={(e) => setConditionType(e.target.value)}
          >
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
          <input
            type={column.isNumeric ? "number" : "text"}
            className="w-full px-2 py-1 border rounded text-sm"
            value={conditionValue}
            onChange={(e) => setConditionValue(e.target.value)}
            placeholder="Wartość..."
          />
        </div>
      )}
      <div className="p-2 border-t flex justify-end space-x-2">
        <button
          className="px-3 py-1 text-sm text-gray-600 hover:bg-gray-100 rounded"
          onClick={onClose}
        >
          Anuluj
        </button>
        <button
          className="px-3 py-1 text-sm text-white bg-blue-600 hover:bg-blue-700 rounded"
          onClick={handleApplyFilter}
        >
          Zastosuj
        </button>
      </div>
    </div>
  );
};

const FilterInput = ({ 
  column, 
  onFilterChange, 
  uniqueValues 
}: { 
  column: ExtendedColumnDefinition; 
  onFilterChange: (columnId: string, filter: ColumnFilter | null) => void;
  uniqueValues: (string | number | boolean)[];
}) => {
  const [isDropdownOpen, setIsDropdownOpen] = React.useState(false);

  return (
    <div className="relative">
      <button
        className="w-full px-2 py-1 text-left text-sm border rounded hover:bg-gray-50"
        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
      >
        {column.filter ? 'Filtr aktywny' : 'Filtruj...'}
      </button>
      {isDropdownOpen && (
        <FilterDropdown
          column={column}
          uniqueValues={uniqueValues}
          onFilterChange={onFilterChange}
          onClose={() => setIsDropdownOpen(false)}
        />
      )}
    </div>
  );
};

const TableHeader: React.FC<TableHeaderProps> = ({
  columns,
  sortField,
  sortDirection,
  onSort,
  onToggleFreeze,
  onFilterChange,
  frozenColumnOffset,
  filteredCount,
  totalCount,
  visibleGroups
}) => {
  // Pobierz unikalne wartości dla każdej kolumny
  const uniqueColumnValues = useMemo(() => {
    const values = new Map<string, Set<string | number | boolean>>();
    
    columns.forEach(column => {
      const columnValues = new Set<string | number | boolean>();
      visibleGroups.forEach(group => {
        const value = column.getValue(group);
        if (value !== undefined && value !== null) {
          columnValues.add(value);
        }
      });
      values.set(column.id, columnValues);
    });
    
    return values;
  }, [columns, visibleGroups]);

  return (
    <thead className="bg-gray-50 sticky top-0 z-30">
      {/* Panel podsumowania filtrów */}
      <tr>
        <th colSpan={columns.length + 2} className="px-6 py-2 bg-white border-b">
          <div className="flex justify-between items-center">
            <div className="text-sm text-gray-500">
              Wyświetlane: <span className="font-medium">{filteredCount}</span> z <span className="font-medium">{totalCount}</span> strategii 
              ({Math.round((filteredCount / totalCount) * 100)}%)
            </div>
            <div></div>
          </div>
        </th>
      </tr>

      {/* Nagłówki kolumn */}
      <tr>
        <th className="w-10 px-3 py-3 sticky left-0 z-20 bg-gray-50">
          <span className="text-xs font-medium text-gray-500 uppercase">Szczegóły</span>
        </th>
        <th className="w-10 px-3 py-3 sticky left-[40px] z-20 bg-gray-50">
          <span className="text-xs font-medium text-gray-500 uppercase">Ukryj</span>
        </th>
        {columns.map((column, index) => (
          <th
            key={column.id}
            className={`
              px-6 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider
              ${column.frozen ? 'sticky z-20' : ''}
              ${column.frozen ? 'shadow-[2px_0_5px_-2px_rgba(0,0,0,0.1)]' : ''}
              ${column.filter ? 'bg-gray-100' : ''}
            `}
            style={column.frozen ? {
              left: `${frozenColumnOffset(index)}px`,
              backgroundColor: column.filter ? 'rgb(243 244 246)' : 'rgb(249, 250, 251)',
              borderRight: '1px solid #e5e7eb',
            } : undefined}
          >
            <div className="flex flex-col space-y-2">
              <div className="flex items-center justify-between">
                <span
                  className="cursor-pointer hover:bg-gray-100 px-2 py-1 rounded"
                  onClick={() => onSort(column.id)}
                >
                  {column.header} {sortField === column.id && (sortDirection === 'asc' ? '↑' : '↓')}
                </span>
                <input
                  type="checkbox"
                  checked={column.frozen}
                  onChange={() => onToggleFreeze(column.id)}
                  className="ml-2"
                />
              </div>
              <FilterInput
                column={column}
                onFilterChange={onFilterChange}
                uniqueValues={Array.from(uniqueColumnValues.get(column.id) || [])}
              />
            </div>
          </th>
        ))}
      </tr>
    </thead>
  );
};

export default TableHeader;
