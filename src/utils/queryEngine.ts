import { ColumnClassification, DataRow } from '../types';

export interface NLFilterResult {
  query: string;
  filteredRows: DataRow[];
  originalCount: number;
  filteredCount: number;
  percentage: number;
  appliedRules: string[];
  summary: string;
}

export function filterDatasetByNL(
  rows: DataRow[],
  colTypes: ColumnClassification,
  userQuery: string
): NLFilterResult {
  if (!rows || rows.length === 0) {
    return {
      query: userQuery,
      filteredRows: [],
      originalCount: 0,
      filteredCount: 0,
      percentage: 0,
      appliedRules: [],
      summary: 'Empty dataset.'
    };
  }

  const originalCount = rows.length;

  if (!userQuery || !userQuery.trim()) {
    return {
      query: '',
      filteredRows: rows,
      originalCount,
      filteredCount: originalCount,
      percentage: 100,
      appliedRules: [],
      summary: `Displaying full dataset (${originalCount.toLocaleString()} rows).`
    };
  }

  const queryRaw = userQuery.trim();
  const queryLower = queryRaw.toLowerCase();
  const numericCols = colTypes.numeric;
  const categoricalCols = [...colTypes.categorical, ...colTypes.text, ...colTypes.id_high_cardinality];

  const appliedRules: string[] = [];
  const filters: Array<(row: DataRow) => boolean> = [];

  // 1. Numeric comparison matching
  // Matches e.g. "sales > 5000", "above 1200", "revenue over 2000", "under 500", "clicks <= 10000"
  const numOpRegex = /(?:([a-zA-Z0-9_\s]+)\s+)?(above|greater than|more than|over|>|>=|exceeding|higher than|below|less than|under|<|<=|lower than|smaller than|equal to|equals|=|==|is)\s*(?:[₹$€£]\s*)?([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)/gi;

  let match: RegExpExecArray | null;
  while ((match = numOpRegex.exec(queryRaw)) !== null) {
    const colPrefix = (match[1] || '').trim().toLowerCase();
    const operator = match[2].toLowerCase();
    const numVal = parseFloat(match[3].replace(/,/g, ''));

    let targetCol: string | null = null;

    if (colPrefix) {
      for (const nc of numericCols) {
        const ncClean = nc.toLowerCase().replace(/_/g, ' ');
        if (ncClean.includes(colPrefix) || colPrefix.includes(ncClean)) {
          targetCol = nc;
          break;
        }
      }
    }

    if (!targetCol) {
      for (const nc of numericCols) {
        const ncClean = nc.toLowerCase().replace(/_/g, ' ');
        if (queryLower.includes(ncClean)) {
          targetCol = nc;
          break;
        }
      }
    }

    if (!targetCol && numericCols.length > 0) {
      targetCol = numericCols[0];
    }

    if (targetCol) {
      const col = targetCol;
      if (['above', 'greater than', 'more than', 'over', '>', '>=', 'exceeding', 'higher than'].includes(operator)) {
        filters.push(r => (Number(r[col]) || 0) > numVal);
        appliedRules.push(`\`${col}\` > ${numVal.toLocaleString()}`);
      } else if (['below', 'less than', 'under', '<', '<=', 'lower than', 'smaller than'].includes(operator)) {
        filters.push(r => (Number(r[col]) || 0) < numVal);
        appliedRules.push(`\`${col}\` < ${numVal.toLocaleString()}`);
      } else {
        filters.push(r => Math.abs((Number(r[col]) || 0) - numVal) < 1e-4);
        appliedRules.push(`\`${col}\` == ${numVal.toLocaleString()}`);
      }
    }
  }

  // 2. Categorical string matching
  // Remove matched numeric tokens to avoid duplicate matching
  let cleanTextQuery = queryRaw.replace(numOpRegex, ' ');
  // Remove filler stop words
  const stopWords = ['show', 'me', 'the', 'rows', 'records', 'where', 'with', 'in', 'all', 'orders', 'data', 'having', 'is', 'are', 'find', 'only'];
  const words = cleanTextQuery
    .toLowerCase()
    .replace(/[^a-z0-9_\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 1 && !stopWords.includes(w));

  for (const word of words) {
    let matchedCol: string | null = null;
    let matchedVal: string | null = null;

    for (const col of categoricalCols) {
      for (const row of rows) {
        const v = String(row[col] ?? '').toLowerCase();
        if (v.includes(word)) {
          matchedCol = col;
          matchedVal = String(row[col]);
          break;
        }
      }
      if (matchedCol) break;
    }

    if (matchedCol && matchedVal) {
      const c = matchedCol;
      const w = word;
      filters.push(r => String(r[c] ?? '').toLowerCase().includes(w));
      appliedRules.push(`\`${c}\` contains "${w}"`);
    }
  }

  let filteredRows = rows;
  if (filters.length > 0) {
    filteredRows = rows.filter(r => filters.every(f => f(r)));
  }

  const filteredCount = filteredRows.length;
  const percentage = Number(((filteredCount / originalCount) * 100).toFixed(1));

  return {
    query: userQuery,
    filteredRows,
    originalCount,
    filteredCount,
    percentage,
    appliedRules,
    summary: appliedRules.length > 0 ? appliedRules.join(' AND ') : `Searched for "${userQuery}"`
  };
}
