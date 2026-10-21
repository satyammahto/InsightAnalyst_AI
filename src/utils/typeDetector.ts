import { ColumnClassification, DataRow } from '../types';

export function detectColumnTypes(rows: DataRow[]): ColumnClassification {
  const result: ColumnClassification = {
    numeric: [],
    categorical: [],
    date: [],
    id_high_cardinality: [],
    text: [],
    details: {},
    telemetry_table: []
  };

  const nRows = rows.length;
  if (nRows === 0) return result;

  const columns = Object.keys(rows[0] || {});

  for (const col of columns) {
    const colLower = col.toLowerCase().trim();
    const values = rows.map(r => r[col]);
    const nonNulls = values.filter(v => v !== null && v !== undefined && String(v).trim() !== '' && String(v).toLowerCase() !== 'nan');
    const nonNullCount = nonNulls.length;
    const nullCount = nRows - nonNullCount;
    const missingPct = (nullCount / Math.max(nRows, 1)) * 100;

    const uniqueSet = new Set(nonNulls.map(v => String(v).trim()));
    const nUnique = uniqueSet.size;
    const uniqueRatio = nUnique / Math.max(nRows, 1);

    let inferredType = 'text';

    // 1. Check if numeric
    const isNumericCandidate = nonNulls.length > 0 && nonNulls.slice(0, 100).every(v => {
      const s = String(v).replace(/,/g, '').trim();
      return !isNaN(Number(s)) && isFinite(Number(s));
    });

    // Check if ID keyword present
    const idKeywords = ['id', 'uuid', 'guid', 'ssn', 'tax_id', 'account_num', 'index', 'key', 'code', 'hash', 'txn', 'cmp', 'pt'];
    const isIdNamed = idKeywords.some(kw => colLower === kw || colLower.endsWith('_' + kw) || colLower.startsWith(kw + '_'));

    // 2. Check if Date
    const isDateNamed = ['date', 'time', 'timestamp', 'dt', 'dob', 'created', 'updated'].some(kw => colLower.includes(kw));
    let isDate = false;

    if (isDateNamed || (!isNumericCandidate && nonNulls.length > 0)) {
      const sample = nonNulls.slice(0, 50);
      let parseSuccess = 0;
      for (const val of sample) {
        const s = String(val).trim();
        if ((s.includes('-') || s.includes('/') || s.includes(':')) && s.length >= 6) {
          const t = Date.parse(s);
          if (!isNaN(t)) {
            parseSuccess++;
          }
        }
      }
      if (sample.length > 0 && parseSuccess / sample.length >= 0.6) {
        isDate = true;
      }
    }

    if (isDate) {
      inferredType = 'date';
    } else if (isIdNamed && (uniqueRatio > 0.3 || nUnique > 20)) {
      inferredType = 'id_high_cardinality';
    } else if (isNumericCandidate) {
      // If unique count is tiny (e.g. 2 or 3) and it's integers like 0/1 or rating, check if it's categorical
      if (nUnique <= 4 && nRows >= 20 && !['revenue', 'sales', 'spend', 'price', 'rating', 'age'].some(k => colLower.includes(k))) {
        inferredType = 'categorical';
      } else {
        inferredType = 'numeric';
      }
    } else if (nUnique <= Math.min(25, nRows * 0.7) || uniqueRatio < 0.25) {
      inferredType = 'categorical';
    } else if (uniqueRatio >= 0.9 && nRows >= 10) {
      inferredType = 'id_high_cardinality';
    } else {
      inferredType = 'text';
    }

    result[inferredType as keyof Pick<ColumnClassification, 'numeric' | 'categorical' | 'date' | 'id_high_cardinality' | 'text'>].push(col);
    result.details[col] = inferredType;

    const sampleSlice = Array.from(uniqueSet).slice(0, 3);
    let sampleStr = sampleSlice.join(', ');
    if (sampleStr.length > 40) sampleStr = sampleStr.slice(0, 37) + '...';

    result.telemetry_table.push({
      featureName: col,
      schemaType: inferredType,
      nonNullCount: nonNullCount.toLocaleString(),
      uniquePct: `${(uniqueRatio * 100).toFixed(1)}%`,
      missingPct: missingPct > 0 ? `${missingPct.toFixed(1)}%` : '0.0%',
      sampleValues: sampleStr
    });
  }

  return result;
}
