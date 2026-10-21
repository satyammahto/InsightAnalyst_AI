"""
DuckDB Analytics Engine for InsightAnalyst AI
----------------------------------------------
High-performance, embedded in-memory SQL analytics engine powered by DuckDB.
Supports:
- Zero-copy DataFrame registration (aliased as 'dataset', 'df', 'data')
- Strict read-only query security validation
- Schema metadata inspection and LLM prompt generation
- Rule-based & LLM-powered natural-language-to-SQL translation
- Automatic Plotly visualization generation from query results
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objects as go


class DuckDBEngine:
    """
    High-performance, embedded in-memory SQL analytics engine.
    """
    FORBIDDEN_KEYWORDS = [
        r'\bDROP\b', r'\bDELETE\b', r'\bINSERT\b', r'\bUPDATE\b',
        r'\bALTER\b', r'\bCREATE\b', r'\bATTACH\b', r'\bDETACH\b',
        r'\bCOPY\b', r'\bEXPORT\b', r'\bIMPORT\b', r'\bINSTALL\b',
        r'\bLOAD\b', r'\bPRAGMA\b', r'\bCALL\b', r'\bEXECUTE\b',
        r'\bSET\b', r'\bVACUUM\b', r'\bCHECKPOINT\b', r'\bREPLACE\b'
    ]

    ALLOWED_PREFIXES = ['SELECT', 'WITH', 'DESCRIBE', 'EXPLAIN', 'SHOW', 'SUMMARIZE']

    def __init__(self, df: pd.DataFrame, table_name: str = "dataset"):
        self.df = df if df is not None else pd.DataFrame()
        self.table_name = table_name
        self.con = duckdb.connect(database=":memory:")
        self._register_dataset()

    def _register_dataset(self):
        """Registers the active pandas dataframe into in-memory DuckDB under multiple aliases."""
        if self.df is not None and not self.df.empty:
            # Register base table
            self.con.register(self.table_name, self.df)
            self.con.register("df", self.df)
            self.con.register("data", self.df)

            # Create clean alias view with normalized column identifiers (e.g. CO_GT instead of CO(GT))
            clean_cols = []
            for col in self.df.columns:
                safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(col)).strip('_')
                if not safe_name or safe_name[0].isdigit():
                    safe_name = f"col_{safe_name}"
                # Handle potential duplicate clean column names
                orig_safe = safe_name
                counter = 1
                while safe_name in clean_cols:
                    safe_name = f"{orig_safe}_{counter}"
                    counter += 1
                clean_cols.append((col, safe_name))

            select_exprs = [f'"{orig}" AS "{clean}"' for orig, clean in clean_cols]
            create_clean_view_sql = f'CREATE OR REPLACE VIEW dataset_clean AS SELECT {", ".join(select_exprs)} FROM "{self.table_name}"'
            try:
                self.con.execute(create_clean_view_sql)
            except Exception:
                pass

    def validate_sql(self, sql_query: str) -> Tuple[bool, Optional[str]]:
        """
        Validates SQL query against security policies:
        - Only read-only analytical queries allowed (SELECT, WITH, DESCRIBE, SHOW, SUMMARIZE, EXPLAIN)
        - DDL, DML, and disk operations strictly rejected
        """
        if not sql_query or not sql_query.strip():
            return False, "Query is empty."

        cleaned = sql_query.strip()
        # Remove SQL comments
        cleaned = re.sub(r'--.*$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL).strip()

        if not cleaned:
            return False, "Query contains only comments or whitespace."

        # Check starting keyword
        first_token = re.split(r'\s+', cleaned)[0].upper()
        if first_token not in self.ALLOWED_PREFIXES:
            return False, f"Forbidden query type '{first_token}'. Only read-only queries (SELECT, WITH, DESCRIBE, SHOW, SUMMARIZE) are allowed."

        # Check for forbidden mutation or administration keywords
        for pattern in self.FORBIDDEN_KEYWORDS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                matched = re.search(pattern, cleaned, re.IGNORECASE).group(0)
                return False, f"Forbidden SQL keyword '{matched}'. Engine operates in strict read-only mode."

        # Reject multiple chained statements with semicolons (e.g., SELECT ...; DROP ...)
        statements = [s.strip() for s in cleaned.split(';') if s.strip()]
        if len(statements) > 1:
            return False, "Chained multi-statement SQL queries are not permitted for security."

        return True, None

    def execute_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Executes a validated SQL query against DuckDB and returns timed results.
        """
        is_valid, error_msg = self.validate_sql(sql_query)
        if not is_valid:
            return {
                "success": False,
                "df": None,
                "row_count": 0,
                "execution_time_ms": 0.0,
                "sql": sql_query,
                "error": error_msg
            }

        start_time = time.perf_counter()
        try:
            res_df = self.con.execute(sql_query).df()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "success": True,
                "df": res_df,
                "row_count": len(res_df),
                "execution_time_ms": round(elapsed_ms, 2),
                "sql": sql_query,
                "error": None
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "success": False,
                "df": None,
                "row_count": 0,
                "execution_time_ms": round(elapsed_ms, 2),
                "sql": sql_query,
                "error": f"SQL Execution Error: {str(e)}"
            }

    def quote_col(self, col_name: str) -> str:
        """Properly quotes a column name for SQL execution."""
        return f'"{col_name}"'

    def get_schema_summary(self) -> List[Dict[str, Any]]:
        """Returns structured metadata for all columns in the dataset."""
        if self.df is None or self.df.empty:
            return []

        summary = []
        for col in self.df.columns:
            s = self.df[col]
            dtype_str = str(s.dtype)
            non_null = int(s.notnull().sum())
            null_count = int(s.isnull().sum())
            unique_count = int(s.nunique(dropna=True))

            # Sample value
            sample_valid = s.dropna()
            sample_val = str(sample_valid.iloc[0]) if len(sample_valid) > 0 else "None"
            if len(sample_val) > 40:
                sample_val = sample_val[:37] + "..."

            summary.append({
                "column_name": col,
                "quoted_name": f'"{col}"',
                "dtype": dtype_str,
                "non_null_count": non_null,
                "null_count": null_count,
                "unique_count": unique_count,
                "sample": sample_val
            })
        return summary

    def get_schema_prompt_context(self) -> str:
        """Returns a concise SQL schema representation for LLM context."""
        if self.df is None or self.df.empty:
            return "Table: dataset (empty)"

        col_defs = []
        for col in self.df.columns:
            s = self.df[col]
            if pd.api.types.is_numeric_dtype(s):
                type_name = "DOUBLE" if pd.api.types.is_float_dtype(s) else "BIGINT"
            elif pd.api.types.is_datetime64_any_dtype(s):
                type_name = "TIMESTAMP"
            elif pd.api.types.is_bool_dtype(s):
                type_name = "BOOLEAN"
            else:
                type_name = "VARCHAR"
            col_defs.append(f'  "{col}" {type_name}')

        cols_str = ",\n".join(col_defs)
        schema_ddl = f"CREATE TABLE dataset (\n{cols_str}\n);\n-- Total records: {len(self.df):,}"
        return schema_ddl

    def is_raw_sql(self, text: str) -> bool:
        """Checks if the user prompt is already a direct SQL query."""
        if not text:
            return False
        tokens = text.strip().split()
        if not tokens:
            return False
        first_word = tokens[0].upper().rstrip(';')
        if first_word in ['SELECT', 'WITH']:
            return True
        if first_word == 'SUMMARIZE' and len(tokens) <= 3:
            return True
        if first_word in ['DESCRIBE', 'EXPLAIN'] and len(tokens) <= 3:
            return True
        if first_word == 'SHOW' and len(tokens) >= 2 and tokens[1].upper().rstrip(';') in ['TABLES', 'COLUMNS', 'DATABASES', 'VIEWS', 'ALL']:
            return True
        return False

    def generate_sample_queries(self) -> List[Dict[str, str]]:
        """Generates contextual, ready-to-run DuckDB SQL queries based on the active dataset."""
        if self.df is None or self.df.empty:
            return []

        queries = []
        numeric_cols = [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]
        cat_cols = [c for c in self.df.columns if not pd.api.types.is_numeric_dtype(self.df[c]) and not pd.api.types.is_datetime64_any_dtype(self.df[c])]
        date_cols = [c for c in self.df.columns if pd.api.types.is_datetime64_any_dtype(self.df[c])]

        # 1. First 10 rows preview
        queries.append({
            "title": "📋 Preview First 10 Records",
            "sql": "SELECT * FROM dataset LIMIT 10;"
        })

        # 2. Dataset Quick Summary using DuckDB SUMMARIZE
        queries.append({
            "title": "📊 Statistical Summary (All Columns)",
            "sql": "SUMMARIZE dataset;"
        })

        # 3. Categorical breakdown
        if cat_cols:
            lead_cat = cat_cols[0]
            queries.append({
                "title": f"🏷️ Value Counts by {lead_cat}",
                "sql": f'SELECT "{lead_cat}", COUNT(*) AS count, ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM dataset), 2) AS percentage\nFROM dataset\nGROUP BY "{lead_cat}"\nORDER BY count DESC\nLIMIT 10;'
            })

        # 4. Top 10 by primary numeric metric
        if numeric_cols:
            lead_num = numeric_cols[0]
            queries.append({
                "title": f"🔝 Top 10 Records with Highest {lead_num}",
                "sql": f'SELECT * FROM dataset\nORDER BY "{lead_num}" DESC NULLS LAST\nLIMIT 10;'
            })

        # 5. Grouped aggregation (Cat x Num)
        if cat_cols and numeric_cols:
            c = cat_cols[0]
            n = numeric_cols[0]
            queries.append({
                "title": f"📈 {n} Aggregation by {c}",
                "sql": f'SELECT "{c}", COUNT(*) AS total_records, ROUND(AVG("{n}"), 2) AS avg_{n}, ROUND(SUM("{n}"), 2) AS total_{n}\nFROM dataset\nGROUP BY "{c}"\nORDER BY total_{n} DESC NULLS LAST\nLIMIT 10;'
            })

        # 6. Date aggregation if available
        if date_cols and numeric_cols:
            d = date_cols[0]
            n = numeric_cols[0]
            queries.append({
                "title": f"📅 Monthly Trend of {n}",
                "sql": f'SELECT DATE_TRUNC(\'month\', CAST("{d}" AS TIMESTAMP)) AS month, ROUND(AVG("{n}"), 2) AS avg_{n}, COUNT(*) AS records\nFROM dataset\nGROUP BY 1\nORDER BY 1 ASC;'
            })

        return queries

    def translate_nl_to_sql_offline(self, nl_query: str) -> Optional[Tuple[str, str]]:
        """
        Deterministic, rule-based natural language to DuckDB SQL translator.
        Handles common questions:
        - "top 10 rows by X", "highest X", "lowest X"
        - "average X by Y", "mean X by Y"
        - "sum/total X by Y"
        - "count by Y", "how many records by Y"
        - "summary statistics", "summarize data"
        """
        if not nl_query or not nl_query.strip():
            return None

        q = nl_query.strip().lower()
        cols = list(self.df.columns)
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self.df[c])]
        cat_cols = [c for c in cols if c not in numeric_cols]

        # Check for SUMMARIZE or DESCRIBE requests
        if any(w in q for w in ['summarize dataset', 'summarize data', 'summary of all columns', 'describe dataset', 'describe table']):
            return "SUMMARIZE dataset;", "Computed full statistical distribution across all columns using DuckDB SUMMARIZE."

        # Extract limit if specified
        limit = 10
        limit_match = re.search(r'\b(?:top|first|limit|best|worst|bottom|highest|lowest)?\s*(\d+)\b', q)
        if limit_match:
            try:
                limit = int(limit_match.group(1))
            except Exception:
                limit = 10

        # Helper to find matched column
        def match_col(query_str: str, candidates: List[str]) -> Optional[str]:
            for col in candidates:
                # Exact lower check
                if col.lower() in query_str:
                    return col
                # Cleaned name check (without underscores/symbols)
                c_clean = re.sub(r'[^a-zA-Z0-9]', ' ', col.lower()).strip()
                if c_clean and c_clean in query_str:
                    return col
            return None

        matched_num = match_col(q, numeric_cols)
        matched_cat = match_col(q, cat_cols)

        # 1. Top / Highest N
        if any(w in q for w in ['highest', 'top', 'maximum', 'max', 'largest', 'peak']) and matched_num:
            sql = f'SELECT * FROM dataset\nORDER BY "{matched_num}" DESC NULLS LAST\nLIMIT {limit};'
            return sql, f"Retrieved top {limit} records with the highest '{matched_num}'."

        # 2. Lowest / Bottom N
        if any(w in q for w in ['lowest', 'bottom', 'minimum', 'min', 'smallest']) and matched_num:
            sql = f'SELECT * FROM dataset\nORDER BY "{matched_num}" ASC NULLS LAST\nLIMIT {limit};'
            return sql, f"Retrieved bottom {limit} records with the lowest '{matched_num}'."

        # 3. Average / Mean by Category
        if any(w in q for w in ['average', 'mean', 'avg']) and matched_num:
            if matched_cat:
                sql = (
                    f'SELECT "{matched_cat}", ROUND(AVG("{matched_num}"), 2) AS avg_{matched_num}, COUNT(*) AS records\n'
                    f'FROM dataset\n'
                    f'GROUP BY "{matched_cat}"\n'
                    f'ORDER BY avg_{matched_num} DESC NULLS LAST\n'
                    f'LIMIT {limit};'
                )
                return sql, f"Calculated average '{matched_num}' grouped by '{matched_cat}'."
            else:
                sql = f'SELECT ROUND(AVG("{matched_num}"), 2) AS avg_{matched_num}, ROUND(MIN("{matched_num}"), 2) AS min_{matched_num}, ROUND(MAX("{matched_num}"), 2) AS max_{matched_num} FROM dataset;'
                return sql, f"Calculated baseline statistics for '{matched_num}'."

        # 4. Total / Sum by Category
        if any(w in q for w in ['total', 'sum']) and matched_num:
            if matched_cat:
                sql = (
                    f'SELECT "{matched_cat}", ROUND(SUM("{matched_num}"), 2) AS total_{matched_num}, COUNT(*) AS records\n'
                    f'FROM dataset\n'
                    f'GROUP BY "{matched_cat}"\n'
                    f'ORDER BY total_{matched_num} DESC NULLS LAST\n'
                    f'LIMIT {limit};'
                )
                return sql, f"Calculated total sum of '{matched_num}' grouped by '{matched_cat}'."
            else:
                sql = f'SELECT ROUND(SUM("{matched_num}"), 2) AS total_{matched_num} FROM dataset;'
                return sql, f"Calculated overall sum for '{matched_num}'."

        # 5. Count / Frequency by Category
        if any(w in q for w in ['count', 'how many', 'distribution of', 'frequency']) and matched_cat:
            sql = (
                f'SELECT "{matched_cat}", COUNT(*) AS count,\n'
                f'       ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM dataset), 2) AS percentage\n'
                f'FROM dataset\n'
                f'GROUP BY "{matched_cat}"\n'
                f'ORDER BY count DESC\n'
                f'LIMIT {limit};'
            )
            return sql, f"Calculated record count and percentage distribution for '{matched_cat}'."

        # 6. Generic Top 10 preview if asking for general records
        if any(w in q for w in ['preview', 'show records', 'first rows', 'show table', 'sample records']):
            return f"SELECT * FROM dataset LIMIT {limit};", f"Displaying first {limit} rows."

        return None

    def translate_nl_to_sql_llm(self, nl_query: str, api_key: str) -> Optional[Tuple[str, str]]:
        """
        Translates complex natural language questions to DuckDB SQL using Google Gemini API.
        """
        if not api_key or len(api_key) < 10:
            return None

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')

            schema_ddl = self.get_schema_prompt_context()

            prompt = (
                "You are an expert DuckDB SQL engineer. Generate a single valid, read-only DuckDB SQL query "
                "to answer the user's analytical question.\n\n"
                f"### DATABASE SCHEMA:\n{schema_ddl}\n\n"
                "### RULES:\n"
                "1. Always query from the table named `dataset`.\n"
                "2. Double-quote all column names that contain spaces or special characters (e.g. \"CO(GT)\", \"NMHC(GT)\").\n"
                "3. Only use read-only analytical SQL: SELECT, WITH, GROUP BY, ORDER BY, LIMIT, aggregations, window functions.\n"
                "4. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or COPY statements.\n"
                "5. Return ONLY the raw SQL query. Do not wrap in markdown quotes if possible, or use standard ```sql code block.\n\n"
                f"User Question: {nl_query}\n"
                "SQL Query:"
            )

            res = model.generate_content(prompt)
            if res and res.text:
                raw_sql = res.text.strip()
                # Clean markdown fencing if returned
                if raw_sql.startswith("```"):
                    raw_sql = re.sub(r'^```(?:sql)?\n?', '', raw_sql)
                    raw_sql = re.sub(r'\n?```$', '', raw_sql)
                raw_sql = raw_sql.strip()

                is_valid, err = self.validate_sql(raw_sql)
                if is_valid:
                    return raw_sql, f"Generated DuckDB SQL via Gemini for: '{nl_query}'"
        except Exception:
            pass

        return None

    def nl_to_sql(self, nl_query: str, api_key: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Primary entry point to convert natural language or direct SQL into an executable query.
        """
        if self.is_raw_sql(nl_query):
            return nl_query.strip(), "Direct SQL Query"

        # 1. Try LLM translation if key available
        if api_key:
            llm_result = self.translate_nl_to_sql_llm(nl_query, api_key)
            if llm_result:
                return llm_result

        # 2. Try rule-based deterministic translation
        rule_result = self.translate_nl_to_sql_offline(nl_query)
        if rule_result:
            return rule_result

        return None, None

    def auto_visualize(self, result_df: pd.DataFrame, title: str = "") -> Optional[go.Figure]:
        """
        Automatically selects and renders the most appropriate Plotly chart for a query result.
        - 1 categorical + 1 numeric: Bar chart
        - 1 datetime + 1-2 numeric: Line/Area trend
        - 2 numeric: Scatter plot
        - 1 categorical + 2 numeric: Grouped Bar chart
        """
        if result_df is None or result_df.empty:
            return None

        # Exclude internal/index columns
        cols = list(result_df.columns)
        if len(cols) < 2 or len(result_df) == 1:
            return None

        # Identify types in the result
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(result_df[c])]
        non_num_cols = [c for c in cols if c not in numeric_cols]

        date_cols = []
        cat_cols = []
        for c in non_num_cols:
            if pd.api.types.is_datetime64_any_dtype(result_df[c]) or 'date' in c.lower() or 'month' in c.lower() or 'day' in c.lower() or 'time' in c.lower():
                date_cols.append(c)
            else:
                cat_cols.append(c)

        chart_title = f"<b>{title}</b>" if title else "<b>DuckDB Query Auto-Visualization</b>"

        try:
            # Case 1: Time Series Trend
            if date_cols and numeric_cols:
                x_col = date_cols[0]
                y_col = numeric_cols[0]
                fig = px.line(
                    result_df,
                    x=x_col,
                    y=y_col,
                    title=chart_title,
                    template="plotly_white",
                    color_discrete_sequence=["#0284C7"]
                )
                fig.update_traces(line=dict(width=2.5))
                fig.update_layout(margin=dict(l=30, r=30, t=50, b=30), height=350)
                return fig

            # Case 2: 1 Categorical + 1 Numeric Bar Chart
            if cat_cols and len(numeric_cols) == 1:
                x_col = cat_cols[0]
                y_col = numeric_cols[0]
                # If too many categories, take top 20
                plot_data = result_df.head(20)
                fig = px.bar(
                    plot_data,
                    x=x_col,
                    y=y_col,
                    title=chart_title,
                    template="plotly_white",
                    color=y_col,
                    color_continuous_scale="Blues"
                )
                fig.update_layout(margin=dict(l=30, r=30, t=50, b=30), height=350)
                return fig

            # Case 3: 1 Categorical + 2 Numeric Grouped Bar Chart
            if cat_cols and len(numeric_cols) >= 2:
                x_col = cat_cols[0]
                y_cols = numeric_cols[:2]
                plot_data = result_df.head(15)
                fig = px.bar(
                    plot_data,
                    x=x_col,
                    y=y_cols,
                    barmode="group",
                    title=chart_title,
                    template="plotly_white",
                    color_discrete_sequence=["#0284C7", "#38BDF8"]
                )
                fig.update_layout(margin=dict(l=30, r=30, t=50, b=30), height=350)
                return fig

            # Case 4: 2 Numeric Scatter Plot
            if len(numeric_cols) >= 2 and len(result_df) > 5:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                fig = px.scatter(
                    result_df.head(500),
                    x=x_col,
                    y=y_col,
                    title=chart_title,
                    template="plotly_white",
                    color_discrete_sequence=["#0284C7"]
                )
                fig.update_layout(margin=dict(l=30, r=30, t=50, b=30), height=350)
                return fig

        except Exception:
            return None

        return None
