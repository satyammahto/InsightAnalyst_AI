import { DataRow } from '../types';

export class CleaningStudio {
  static imputeMissing(
    rows: DataRow[],
    column: string,
    strategy: string = 'median',
    constantValue?: any
  ): { newRows: DataRow[]; details: string } {
    const newRows = rows.map(r => ({ ...r }));
    const rowsBefore = newRows.length;
    let nullCountBefore = 0;

    for (const r of newRows) {
      const v = r[column];
      if (v === null || v === undefined || String(v).trim() === '' || String(v).toLowerCase() === 'nan') {
        nullCountBefore++;
      }
    }

    let details = '';

    if (strategy === 'drop_rows') {
      const filtered = newRows.filter(r => {
        const v = r[column];
        return v !== null && v !== undefined && String(v).trim() !== '' && String(v).toLowerCase() !== 'nan';
      });
      details = `Dropped ${(rowsBefore - filtered.length).toLocaleString()} rows with missing '${column}'.`;
      return { newRows: filtered, details };
    }

    if (strategy === 'mean') {
      const nums = newRows
        .map(r => Number(r[column]))
        .filter(v => !isNaN(v) && isFinite(v));
      const mean = nums.length > 0 ? nums.reduce((a, b) => a + b, 0) / nums.length : 0;
      for (const r of newRows) {
        const v = r[column];
        if (v === null || v === undefined || String(v).trim() === '' || isNaN(Number(v))) {
          r[column] = Number(mean.toFixed(2));
        }
      }
      details = `Filled ${nullCountBefore.toLocaleString()} missing values with mean (${mean.toFixed(2)}).`;
    } else if (strategy === 'median') {
      const nums = newRows
        .map(r => Number(r[column]))
        .filter(v => !isNaN(v) && isFinite(v))
        .sort((a, b) => a - b);
      const mid = Math.floor(nums.length / 2);
      const median = nums.length > 0 ? (nums.length % 2 !== 0 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2) : 0;
      for (const r of newRows) {
        const v = r[column];
        if (v === null || v === undefined || String(v).trim() === '' || isNaN(Number(v))) {
          r[column] = Number(median.toFixed(2));
        }
      }
      details = `Filled ${nullCountBefore.toLocaleString()} missing values with median (${median.toFixed(2)}).`;
    } else if (strategy === 'mode') {
      const counts: Record<string, number> = {};
      for (const r of newRows) {
        const v = r[column];
        if (v !== null && v !== undefined && String(v).trim() !== '') {
          const s = String(v);
          counts[s] = (counts[s] || 0) + 1;
        }
      }
      let mode = 'Unknown';
      let maxC = 0;
      for (const [k, count] of Object.entries(counts)) {
        if (count > maxC) {
          maxC = count;
          mode = k;
        }
      }
      for (const r of newRows) {
        const v = r[column];
        if (v === null || v === undefined || String(v).trim() === '' || String(v).toLowerCase() === 'nan') {
          r[column] = mode;
        }
      }
      details = `Filled ${nullCountBefore.toLocaleString()} missing values with mode ('${mode}').`;
    } else if (strategy === 'constant') {
      const cVal = constantValue !== undefined && constantValue !== '' ? constantValue : '0';
      for (const r of newRows) {
        const v = r[column];
        if (v === null || v === undefined || String(v).trim() === '' || String(v).toLowerCase() === 'nan') {
          r[column] = cVal;
        }
      }
      details = `Filled ${nullCountBefore.toLocaleString()} missing values with constant value '${cVal}'.`;
    } else if (strategy === 'ffill') {
      let lastVal: any = null;
      for (const r of newRows) {
        const v = r[column];
        if (v !== null && v !== undefined && String(v).trim() !== '') {
          lastVal = v;
        } else if (lastVal !== null) {
          r[column] = lastVal;
        }
      }
      details = `Filled missing values using forward fill.`;
    }

    return { newRows, details };
  }

  static handleOutliers(
    rows: DataRow[],
    column: string,
    method: string = 'clip_percentiles'
  ): { newRows: DataRow[]; details: string } {
    const newRows = rows.map(r => ({ ...r }));
    const nums = newRows
      .map(r => Number(r[column]))
      .filter(v => !isNaN(v) && isFinite(v))
      .sort((a, b) => a - b);

    if (nums.length === 0) return { newRows, details: 'No numeric values found.' };

    let details = '';

    if (method === 'clip_percentiles') {
      const p1Idx = Math.floor(nums.length * 0.01);
      const p99Idx = Math.min(nums.length - 1, Math.floor(nums.length * 0.99));
      const lowVal = nums[p1Idx];
      const highVal = nums[p99Idx];
      let modified = 0;

      for (const r of newRows) {
        const v = Number(r[column]);
        if (!isNaN(v)) {
          if (v < lowVal) {
            r[column] = lowVal;
            modified++;
          } else if (v > highVal) {
            r[column] = highVal;
            modified++;
          }
        }
      }
      details = `Clipped ${modified.toLocaleString()} values to range [${lowVal.toFixed(2)}, ${highVal.toFixed(2)}] (P1-P99).`;
    } else if (method === 'clip_iqr' || method === 'drop_iqr') {
      const q1Idx = Math.floor(nums.length * 0.25);
      const q3Idx = Math.floor(nums.length * 0.75);
      const q1 = nums[q1Idx];
      const q3 = nums[q3Idx];
      const iqr = q3 - q1;
      const lowerBound = q1 - 1.5 * iqr;
      const upperBound = q3 + 1.5 * iqr;

      if (method === 'clip_iqr') {
        let modified = 0;
        for (const r of newRows) {
          const v = Number(r[column]);
          if (!isNaN(v)) {
            if (v < lowerBound) {
              r[column] = Number(lowerBound.toFixed(2));
              modified++;
            } else if (v > upperBound) {
              r[column] = Number(upperBound.toFixed(2));
              modified++;
            }
          }
        }
        details = `Clipped ${modified.toLocaleString()} values outside 1.5*IQR bound [${lowerBound.toFixed(2)}, ${upperBound.toFixed(2)}].`;
      } else {
        const filtered = newRows.filter(r => {
          const v = Number(r[column]);
          return isNaN(v) || (v >= lowerBound && v <= upperBound);
        });
        details = `Dropped ${(newRows.length - filtered.length).toLocaleString()} rows outside 1.5*IQR bounds.`;
        return { newRows: filtered, details };
      }
    } else if (method === 'drop_3sigma') {
      const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
      const std = Math.sqrt(nums.reduce((s, v) => s + (v - mean) * (v - mean), 0) / Math.max(nums.length - 1, 1));
      const lowerBound = mean - 3 * std;
      const upperBound = mean + 3 * std;

      const filtered = newRows.filter(r => {
        const v = Number(r[column]);
        return isNaN(v) || (v >= lowerBound && v <= upperBound);
      });
      details = `Dropped ${(newRows.length - filtered.length).toLocaleString()} rows beyond 3σ bounds [${lowerBound.toFixed(2)}, ${upperBound.toFixed(2)}].`;
      return { newRows: filtered, details };
    }

    return { newRows, details };
  }

  static cleanTextColumn(
    rows: DataRow[],
    column: string,
    casing: string = 'title',
    stripWhitespace: boolean = true,
    removeSpecialChars: boolean = false,
    findReplace?: [string, string] | null
  ): { newRows: DataRow[]; details: string } {
    const newRows = rows.map(r => ({ ...r }));
    let count = 0;

    for (const r of newRows) {
      let val = String(r[column] ?? '');
      if (stripWhitespace) val = val.trim().replace(/\s+/g, ' ');
      if (removeSpecialChars) val = val.replace(/[^a-zA-Z0-9\s]/g, '');

      if (findReplace && findReplace[0]) {
        val = val.split(findReplace[0]).join(findReplace[1] || '');
      }

      if (casing === 'title') {
        val = val
          .split(' ')
          .map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
          .join(' ');
      } else if (casing === 'lower') {
        val = val.toLowerCase();
      } else if (casing === 'upper') {
        val = val.toUpperCase();
      }

      if (val !== String(r[column] ?? '')) {
        r[column] = val;
        count++;
      }
    }

    const details = `Standardized text in '${column}': casing=${casing}, strip=${stripWhitespace}. Modified ${count.toLocaleString()} values.`;
    return { newRows, details };
  }

  static extractDateFeatures(
    rows: DataRow[],
    dateColumn: string,
    features: string[]
  ): { newRows: DataRow[]; details: string } {
    const newRows = rows.map(r => ({ ...r }));

    for (const r of newRows) {
      const v = r[dateColumn];
      const parsed = Date.parse(String(v));
      const d = !isNaN(parsed) ? new Date(parsed) : null;

      if (d) {
        if (features.includes('year')) r[`${dateColumn}_year`] = d.getUTCFullYear();
        if (features.includes('month')) r[`${dateColumn}_month`] = d.getUTCMonth() + 1;
        if (features.includes('day')) r[`${dateColumn}_day`] = d.getUTCDate();
        if (features.includes('dayofweek')) r[`${dateColumn}_dayofweek`] = d.getUTCDay();
        if (features.includes('quarter')) r[`${dateColumn}_quarter`] = Math.floor(d.getUTCMonth() / 3) + 1;
        if (features.includes('is_weekend')) r[`${dateColumn}_is_weekend`] = (d.getUTCDay() === 0 || d.getUTCDay() === 6) ? 1 : 0;
      }
    }

    const details = `Extracted ${features.join(', ')} from '${dateColumn}'.`;
    return { newRows, details };
  }

  static scaleNumeric(
    rows: DataRow[],
    column: string,
    method: 'minmax' | 'standard_zscore' | 'robust'
  ): { newRows: DataRow[]; details: string } {
    const newRows = rows.map(r => ({ ...r }));
    const nums = newRows.map(r => Number(r[column])).filter(v => !isNaN(v));
    const newCol = `${column}_scaled`;

    if (method === 'minmax') {
      const min = Math.min(...nums);
      const max = Math.max(...nums);
      const range = max - min || 1;
      for (const r of newRows) {
        const v = Number(r[column]);
        r[newCol] = !isNaN(v) ? Number(((v - min) / range).toFixed(4)) : 0;
      }
    } else if (method === 'standard_zscore') {
      const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
      const std = Math.sqrt(nums.reduce((s, v) => s + (v - mean) * (v - mean), 0) / Math.max(nums.length - 1, 1)) || 1;
      for (const r of newRows) {
        const v = Number(r[column]);
        r[newCol] = !isNaN(v) ? Number(((v - mean) / std).toFixed(4)) : 0;
      }
    }

    const details = `Generated scaled column '${newCol}' using ${method}.`;
    return { newRows, details };
  }

  static deriveColumn(
    rows: DataRow[],
    newColName: string,
    colA: string,
    operation: string,
    colBOrScalar: string | number
  ): { newRows: DataRow[]; details: string } {
    const newRows = rows.map(r => ({ ...r }));
    const isColB = typeof colBOrScalar === 'string' && colBOrScalar in (newRows[0] || {});

    for (const r of newRows) {
      const valA = Number(r[colA]) || 0;
      const valB = isColB ? (Number(r[colBOrScalar]) || 0) : Number(colBOrScalar);

      if (operation === 'add') {
        r[newColName] = Number((valA + valB).toFixed(2));
      } else if (operation === 'subtract') {
        r[newColName] = Number((valA - valB).toFixed(2));
      } else if (operation === 'multiply') {
        r[newColName] = Number((valA * valB).toFixed(2));
      } else if (operation === 'divide') {
        r[newColName] = valB !== 0 ? Number((valA / valB).toFixed(4)) : 0;
      } else if (operation === 'ratio_pct') {
        r[newColName] = valB !== 0 ? Number(((valA / valB) * 100).toFixed(2)) : 0;
      } else if (operation === 'concat') {
        const strB = isColB ? String(r[colBOrScalar] ?? '') : String(colBOrScalar);
        r[newColName] = `${r[colA]}_${strB}`;
      }
    }

    const details = `Created '${newColName}' = '${colA}' ${operation} '${colBOrScalar}'.`;
    return { newRows, details };
  }

  static dropColumns(
    rows: DataRow[],
    colsToDrop: string[]
  ): { newRows: DataRow[]; details: string } {
    const newRows = rows.map(r => {
      const copy = { ...r };
      for (const col of colsToDrop) {
        delete copy[col];
      }
      return copy;
    });

    const details = `Removed ${colsToDrop.length} column(s): ${colsToDrop.join(', ')}.`;
    return { newRows, details };
  }
}
