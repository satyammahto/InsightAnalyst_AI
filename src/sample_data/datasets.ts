export interface PracticeDataset {
  id: string;
  title: string;
  filename: string;
  description: string;
  csvContent: string;
}

export const PRACTICE_DATASETS: PracticeDataset[] = [
  {
    id: 'sales',
    title: 'Retail Sales Data',
    filename: 'sample_sales.csv',
    description: 'E-commerce sales transactions with revenue, region, quantity, and ratings',
    csvContent: `Transaction_ID,Date,Region,Product_Category,Revenue,Quantity,Customer_Rating
TXN-1001,2025-01-05, North ,Electronics,1200.50,3,4.5
TXN-1002,2025-01-06,South,Clothing,250.00,5,4.0
TXN-1003,01/07/2025,NORTH,Electronics,,2,4.8
TXN-1004,2025-01-08,West,Home & Kitchen,450.75,1,3.9
TXN-1005,2025-01-09,south,Clothing,250.00,5,4.0
TXN-1005,2025-01-09,south,Clothing,250.00,5,4.0
TXN-1006,2025-01-10, East ,Electronics,3100.00,10,5.0
TXN-1007,2025-01-12,WEST,Home & Kitchen,89.99,,3.5
TXN-1008,2025-01-15,North,Clothing,180.20,4,4.1
TXN-1009,2025-01-18,East,Electronics,2200.00,6,4.6
TXN-1010,2025-01-20,North,Home & Kitchen,530.00,2,4.2
TXN-1011,2025-01-22,South,Electronics,1450.00,4,4.4
TXN-1012,2025-01-25,East,Clothing,320.00,3,3.8
TXN-1013,2025-01-28,West,Electronics,4200.00,12,4.9
TXN-1014,2025-02-01,North,Clothing,150.00,2,4.0
TXN-1015,2025-02-03,South,Home & Kitchen,610.00,3,4.3
TXN-1016,2025-02-05,East,Electronics,2800.00,7,4.7
TXN-1017,2025-02-08,West,Clothing,210.00,4,3.9
TXN-1018,2025-02-10,North,Electronics,1900.00,5,4.5
TXN-1019,2025-02-12,South,Home & Kitchen,720.00,4,4.1
TXN-2000,2025-02-15,East,Electronics,15000.00,50,5.0`
  },
  {
    id: 'healthcare',
    title: 'Healthcare Patient Records',
    filename: 'patient_clinical_records.csv',
    description: 'Clinical admissions tracking patient age, blood pressure, cholesterol, and hospital stays',
    csvContent: `Patient_ID,Admission_Date,Age,Blood_Pressure,Cholesterol,Diagnosis,Hospital_Days
PT-801,2025-03-01,65,140,240,Cardiology,5
PT-802,2025-03-02,42,120,180,General,2
PT-803,2025-03-02,42,120,180,General,2
PT-804,2025-03-03,78,160,,Cardiology,12
PT-805,2025-03-04,31,115,165,Orthopedics,1
PT-806,2025-03-05,55,135,210,Neurology,4
PT-807,2025-03-06,62,145,255,Cardiology,7
PT-808,03/07/2025,28,110,150,General,1
PT-809,2025-03-08,84,175,290,Cardiology,15
PT-810,2025-03-09,49,128,195,Neurology,3
PT-811,2025-03-10,53,132,205,Orthopedics,3
PT-812,2025-03-11,70,155,270,Cardiology,9
PT-813,2025-03-12,36,118,172,General,2
PT-814,2025-03-14,60,138,225,Neurology,5
PT-815,2025-03-15,25,108,145,General,1`
  },
  {
    id: 'marketing',
    title: 'Marketing Performance',
    filename: 'marketing_attribution.csv',
    description: 'Multi-channel digital marketing campaigns with spend, clicks, conversions, and ROI',
    csvContent: `Campaign_ID,Launch_Date,Channel,Spend_USD,Clicks,Conversions,ROI_Percent
CMP-01,2025-04-01,Social Media,5000,12000,450,180.5
CMP-02,2025-04-02,Search Ads,8500,25000,920,210.0
CMP-03,2025-04-03,Email,1200,4500,180,250.2
CMP-04,2025-04-04,Social Media,5000,12000,450,180.5
CMP-05,2025-04-05,Influencer,15000,38000,1100,140.0
CMP-06,2025-04-06,Search Ads,9200,28000,1050,225.0
CMP-07,2025-04-08,Video Ads,12500,42000,880,115.8
CMP-08,2025-04-10,Email,1500,5200,210,265.0
CMP-09,2025-04-12,Social Media,6200,15000,560,192.0
CMP-10,2025-04-15,Search Ads,11000,31000,1200,230.0
CMP-11,2025-04-18,Influencer,20000,50000,1400,135.0
CMP-12,2025-04-20,Video Ads,14000,46000,950,120.0`
  }
];
