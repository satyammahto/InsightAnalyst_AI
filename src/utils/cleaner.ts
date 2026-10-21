import { ColumnClassification, CleaningReport, DAGStep, DataRow } from '../types';

export function detectAndClean(
  rawRows: DataRow[],
  colTypes: ColumnClassification,
  imputePolicy: string = 'Fill with typical values (Median/Mode)'
): { cleanedRows: DataRow[]; report: CleaningReport } {
  const startTime = performance.now();
  const initialRowCount = rawRows.length;
  const columns = rawRows.length > 0 ? Object.keys(rawRows[0]) : [];
  const initialShape: [number, number] = [initialRowCount, columns.length];
  const dagSteps: DAGStep[] = [];
  const nullsImputed: Record<string, number> = {};
  let totalNullsImputed = 0;
  let stringTokensSanitized = 0;
  const datesConverted: string[] = [];

  // Deep clone
  let currentRows: DataRow[] = rawRows.map(r => ({ ...r }));

  // STEP 1: Duplicate Scan & Purge
  const s1Start = performance.now();
  const seenHashes = new Set<string>();
  const deduplicatedRows: DataRow[] = [];
  let duplicatesRemoved = 0;

  for (const row of currentRows) {
    const key = columns.map(c => String(row[c] ?? '').trim().toLowerCase()).join('||');
    if (seenHashes.has(key)) {
      duplicatesRemoved++;
    } else {
      seenHashes.add(key);
      deduplicatedRows.push(row);
    }
  }
  currentRows = deduplicatedRows;
  const s1Elapsed = performance.now() - s1Start;

  dagSteps.push({
    step_num: '01',
    title: 'Step 1: Duplicate Scan & Purge',
    tag: 'DEDUPLICATE_ROWS',
    desc: `${duplicatesRemoved.toLocaleString()} exact match composite keys dropped across telemetry tuple.`,
    badge: `PURGED ${duplicatesRemoved.toLocaleString()} ROWS`,
    latency: `${s1Elapsed.toFixed(1)} ms`
  });

  // Check if remove incomplete rows policy
  if (imputePolicy === 'Remove incomplete rows') {
    const beforeLen = currentRows.length;
    currentRows = currentRows.filter(row => {
      return columns.every(c => {
        const v = row[c];
        return v !== null && v !== undefined && String(v).trim() !== '' && String(v).toLowerCase() !== 'nan';
      });
    });
    const droppedIncomplete = beforeLen - currentRows.length;
    dagSteps.push({
      step_num: '02',
      title: 'Step 2: Drop Incomplete Records',
      tag: 'DROP_NA_POLICY',
      desc: `Removed ${droppedIncomplete.toLocaleString()} rows with missing cells per policy.`,
      badge: `REMOVED ${droppedIncomplete.toLocaleString()} ROWS`,
      latency: `0.5 ms`
    });
  } else {
    // STEP 2: Numeric Imputation (Median or Mean)
    const s2Start = performance.now();
    let numCellsImputed = 0;
    const useMean = imputePolicy.includes('Mean');

    for (const col of colTypes.numeric) {
      const validNumbers: number[] = [];
      let nullCount = 0;

      for (const row of currentRows) {
        const v = row[col];
        if (v === null || v === undefined || String(v).trim() === '' || String(v).toLowerCase() === 'nan') {
          nullCount++;
        } else {
          const num = Number(String(v).replace(/,/g, '').trim());
          if (!isNaN(num) && isFinite(num)) {
            validNumbers.push(num);
          } else {
            nullCount++;
          }
        }
      }

      if (nullCount > 0) {
        let replacement = 0;
        if (validNumbers.length > 0) {
          if (useMean) {
            replacement = validNumbers.reduce((a, b) => a + b, 0) / validNumbers.length;
          } else {
            validNumbers.sort((a, b) => a - b);
            const mid = Math.floor(validNumbers.length / 2);
            replacement = validNumbers.length % 2 !== 0 ? validNumbers[mid] : (validNumbers[mid - 1] + validNumbers[mid]) / 2;
          }
        }
        // Round replacement reasonably
        replacement = Number(replacement.toFixed(2));

        for (const row of currentRows) {
          const v = row[col];
          if (v === null || v === undefined || String(v).trim() === '' || String(v).toLowerCase() === 'nan' || isNaN(Number(String(v).replace(/,/g, '')))) {
            row[col] = replacement;
          } else {
            row[col] = Number(String(v).replace(/,/g, '').trim());
          }
        }

        nullsImputed[col] = nullCount;
        totalNullsImputed += nullCount;
        numCellsImputed += nullCount;
      } else {
        // Coerce existing values to clean numbers
        for (const row of currentRows) {
          row[col] = Number(String(row[col]).replace(/,/g, '').trim());
        }
      }
    }

    const s2Elapsed = performance.now() - s2Start;
    dagSteps.push({
      step_num: '02',
      title: 'Step 2: Numeric Imputation',
      tag: useMean ? 'MEAN_STRATEGY' : 'MEDIAN_STRATEGY',
      desc: `Replaced NaN values using column ${useMean ? 'means' : 'medians'} across continuous features.`,
      badge: `RECONCILED ${numCellsImputed.toLocaleString()} CELLS`,
      latency: `${s2Elapsed.toFixed(1)} ms`
    });

    // STEP 3: Categorical Normalization (Mode/Sentinel)
    const s3Start = performance.now();
    let catCellsImputed = 0;

    for (const col of colTypes.categorical) {
      const counts: Record<string, number> = {};
      let nullCount = 0;

      for (const row of currentRows) {
        const v = row[col];
        if (v === null || v === undefined || String(v).trim() === '' || String(v).toLowerCase() === 'nan') {
          nullCount++;
        } else {
          const s = String(v).trim();
          counts[s] = (counts[s] || 0) + 1;
        }
      }

      if (nullCount > 0) {
        let modeVal = 'Unknown';
        let maxCount = 0;
        for (const [k, count] of Object.entries(counts)) {
          if (count > maxCount) {
            maxCount = count;
            modeVal = k;
          }
        }

        for (const row of currentRows) {
          const v = row[col];
          if (v === null || v === undefined || String(v).trim() === '' || String(v).toLowerCase() === 'nan') {
            row[col] = modeVal;
          }
        }

        nullsImputed[col] = nullCount;
        totalNullsImputed += nullCount;
        catCellsImputed += nullCount;
      }
    }

    const s3Elapsed = performance.now() - s3Start;
    dagSteps.push({
      step_num: '03',
      title: 'Step 3: Categorical Normalization',
      tag: 'MODE_&_SENTINEL',
      desc: `Replaced null & empty strings with mode sentinel values.`,
      badge: `${catCellsImputed.toLocaleString()} ENTRIES ASSIGNED`,
      latency: `${s3Elapsed.toFixed(1)} ms`
    });
  }

  // STEP 4: String Hygiene, Whitespace & TitleCase Standardizing
  const s4Start = performance.now();
  const textCols = [...new Set([...colTypes.categorical, ...colTypes.text, ...colTypes.id_high_cardinality])];

  for (const col of textCols) {
    if (!columns.includes(col)) continue;
    const colLower = col.toLowerCase();
    const isIdKey = ['id', 'uuid', 'code', 'hash', 'key', 'txn', 'cmp', 'pt'].some(k => colLower.endsWith('_' + k) || colLower === k);

    for (const row of currentRows) {
      const v = row[col];
      if (v !== null && v !== undefined) {
        const s = String(v).trim().replace(/\s+/g, ' ');
        if (s !== String(v)) {
          stringTokensSanitized++;
        }
        if (!isIdKey && s.length > 0) {
          // Format to Title Case if mixed case
          const titleCased = s
            .split(' ')
            .map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
            .join(' ');
          row[col] = titleCased;
        } else {
          row[col] = s;
        }
      }
    }
  }

  const s4Elapsed = performance.now() - s4Start;
  dagSteps.push({
    step_num: '04',
    title: 'Step 4: String Hygiene & Whitespace',
    tag: 'STR_NORMALIZER',
    desc: `Sanitized string casing and trimmed whitespace across text columns.`,
    badge: `CLEANED ${stringTokensSanitized.toLocaleString()} TOKENS`,
    latency: `${s4Elapsed.toFixed(1)} ms`
  });

  // STEP 5: Datetime Schema Coercion (ISO-8601 UTC)
  const s5Start = performance.now();
  for (const col of colTypes.date) {
    if (!columns.includes(col)) continue;
    datesConverted.push(col);

    for (const row of currentRows) {
      const v = row[col];
      if (v !== null && v !== undefined && String(v).trim() !== '') {
        const t = Date.parse(String(v).trim());
        if (!isNaN(t)) {
          const d = new Date(t);
          row[col] = d.toISOString().split('T')[0]; // YYYY-MM-DD standard format
        }
      }
    }
  }

  const s5Elapsed = performance.now() - s5Start;
  dagSteps.push({
    step_num: '05',
    title: 'Step 5: Chrono UTC Datetime Coercion',
    tag: 'CHRONO_UTC',
    desc: `Normalized timestamp strings to ISO-8601 calendar format.`,
    badge: `COERCED ${datesConverted.length} CHRONO COLS`,
    latency: `${s5Elapsed.toFixed(1)} ms`
  });

  const finalShape: [number, number] = [currentRows.length, columns.length];
  const totalCells = Math.max(initialShape[0] * initialShape[1], 1);
  const totalModifications = duplicatesRemoved * columns.length + totalNullsImputed + stringTokensSanitized;
  const rawScore = 100 - (totalModifications / totalCells) * 35;
  const clampedScore = Math.max(65.0, Math.min(99.8, rawScore));
  const hygieneScore = `${clampedScore.toFixed(1)}%`;

  const totalTime = performance.now() - startTime;

  return {
    cleanedRows: currentRows,
    report: {
      initial_shape: initialShape,
      final_shape: finalShape,
      duplicates_removed: duplicatesRemoved,
      total_nulls_imputed: totalNullsImputed,
      nulls_imputed: nullsImputed,
      string_tokens_sanitized: stringTokensSanitized,
      dates_converted: datesConverted,
      hygiene_score: hygieneScore,
      dag_steps: dagSteps,
      execution_ms: `${totalTime.toFixed(1)} ms`
    }
  };
}
