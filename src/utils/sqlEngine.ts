import alasql from 'alasql';
import { DataRow } from '../types';

export interface SQLQueryResult {
  success: boolean;
  rows: DataRow[];
  rowCount: number;
  executionTimeMs: number;
  error?: string;
}

export class SQLEngine {
  private rows: DataRow[] = [];

  constructor(rows: DataRow[]) {
    this.init(rows);
  }

  init(rows: DataRow[]) {
    this.rows = rows;
    try {
      alasql('DROP TABLE IF EXISTS dataset');
      alasql('DROP TABLE IF EXISTS df');
      alasql('DROP TABLE IF EXISTS data');
      alasql('CREATE TABLE dataset');
      alasql.tables.dataset.data = rows.map(r => ({ ...r }));
      alasql('CREATE TABLE df');
      alasql.tables.df.data = alasql.tables.dataset.data;
      alasql('CREATE TABLE data');
      alasql.tables.data.data = alasql.tables.dataset.data;
    } catch (e) {
      console.error('Failed to init SQL tables:', e);
    }
  }

  executeQuery(query: string): SQLQueryResult {
    const start = performance.now();
    try {
      let q = query.trim();
      if (q.endsWith(';')) q = q.slice(0, -1);

      const res = alasql(q);
      const executionTimeMs = performance.now() - start;
      const rows = Array.isArray(res) ? res : [res];

      return {
        success: true,
        rows,
        rowCount: rows.length,
        executionTimeMs
      };
    } catch (err: any) {
      return {
        success: false,
        rows: [],
        rowCount: 0,
        executionTimeMs: performance.now() - start,
        error: err.message || 'SQL execution failed'
      };
    }
  }

  generateSampleQueries(columns: string[], numericCols: string[], catCols: string[]): Array<{ title: string; sql: string }> {
    const queries: Array<{ title: string; sql: string }> = [
      {
        title: '📋 First 10 Records',
        sql: 'SELECT * FROM dataset LIMIT 10;'
      }
    ];

    if (catCols.length > 0) {
      const cat = catCols[0];
      queries.push({
        title: `📊 Distribution by ${cat}`,
        sql: `SELECT [${cat}], COUNT(*) AS record_count FROM dataset GROUP BY [${cat}] ORDER BY record_count DESC;`
      });
    }

    if (catCols.length > 0 && numericCols.length > 0) {
      const cat = catCols[0];
      const num = numericCols[0];
      queries.push({
        title: `💰 Top ${cat} by Total ${num}`,
        sql: `SELECT [${cat}], SUM([${num}]) AS total_${num.toLowerCase()}, AVG([${num}]) AS avg_${num.toLowerCase()} FROM dataset GROUP BY [${cat}] ORDER BY total_${num.toLowerCase()} DESC LIMIT 5;`
      });
    }

    if (numericCols.length >= 2) {
      const n1 = numericCols[0];
      const n2 = numericCols[1];
      queries.push({
        title: `📈 Statistical Summary for ${n1}`,
        sql: `SELECT MIN([${n1}]) AS min_val, MAX([${n1}]) AS max_val, AVG([${n1}]) AS avg_val, AVG([${n2}]) AS avg_${n2.toLowerCase()} FROM dataset;`
      });
    }

    return queries;
  }
}
