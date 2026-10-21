export const COMMON_COLUMN_MAP: Record<string, string> = {
  ORDERNUMBER: 'Order Number',
  QUANTITYORDERED: 'Quantity Ordered',
  PRICEEACH: 'Price',
  ORDERLINENUMBER: 'Order Line Number',
  SALES: 'Sales',
  ORDERDATE: 'Order Date',
  STATUS: 'Status',
  QTR_ID: 'Quarter',
  MONTH_ID: 'Month',
  YEAR_ID: 'Year',
  PRODUCTLINE: 'Product Line',
  MSRP: 'MSRP',
  PRODUCTCODE: 'Product Code',
  CUSTOMERNAME: 'Customer Name',
  PHONE: 'Phone',
  ADDRESSLINE1: 'Address Line 1',
  ADDRESSLINE2: 'Address Line 2',
  CITY: 'City',
  STATE: 'State',
  POSTALCODE: 'Postal Code',
  COUNTRY: 'Country',
  TERRITORY: 'Territory',
  CONTACTLASTNAME: 'Contact Last Name',
  CONTACTFIRSTNAME: 'Contact First Name',
  DEALSIZE: 'Deal Size',
  TRANSACTION_ID: 'Transaction ID',
  PATIENT_ID: 'Patient ID',
  CAMPAIGN_ID: 'Campaign ID',
  BLOOD_PRESSURE: 'Blood Pressure',
  HOSPITAL_DAYS: 'Hospital Days',
  SPEND_USD: 'Spend (USD)',
  ROI_PERCENT: 'ROI (%)',
  CUSTOMER_RATING: 'Customer Rating',
  PRODUCT_CATEGORY: 'Product Category',
  LAUNCH_DATE: 'Launch Date',
  ADMISSION_DATE: 'Admission Date'
};

export function cleanName(name: string): string {
  if (!name) return '';
  const upper = String(name).trim().toUpperCase();
  if (COMMON_COLUMN_MAP[upper]) {
    return COMMON_COLUMN_MAP[upper];
  }
  const words = String(name).replace(/_/g, ' ').split(/\s+/);
  const acronyms = new Set(['ID', 'MSRP', 'URL', 'USD', 'EUR', 'APAC', 'EMEA', 'NA', 'SKU', 'QTR', 'ZIP', 'ROI', 'TXN', 'CMP', 'PT']);
  
  return words
    .filter(Boolean)
    .map(w => {
      const u = w.toUpperCase();
      if (acronyms.has(u)) return u;
      return w.charAt(0).toUpperCase() + w.slice(1).toLowerCase();
    })
    .join(' ');
}
