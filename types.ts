import type React from 'react';

export interface NavItem {
  id: 'analysis' | 'news' | 'valuation' | 'reports';
  label: string;
  icon: React.FC<React.SVGProps<SVGSVGElement>>;
}

export interface FinancialStatementItem {
  [item: string]: number | string;
}

export interface FinancialStatement {
  [year: string]: FinancialStatementItem;
}

export interface Ratio {
  name: string;
  value: string;
  commentary: string;
  benchmark: string;
}

export interface RatioDataPoint {
  year: string;
  value: number;
}

export interface RatioHistory {
  name: string;
  history: RatioDataPoint[];
}

export interface NewsArticle {
  headline: string;
  source: string;
  summary: string;
}

export interface CompanyData {
  companyName: string;
  ticker: string;
  currency: string;
  summary: string;
  incomeStatement: FinancialStatement;
  balanceSheet: FinancialStatement;
  cashFlowStatement: FinancialStatement;
  ratios: Ratio[];
  ratioHistory: RatioHistory[];
  news: NewsArticle[];
  valuationAssumptions: ValuationAssumptions;
}

export interface ValuationAssumptions {
  revenueGrowthRate: number;
  ebitdaMargin: number;
  taxRate: number;
  capexAsPercentageOfRevenue: number;
  depreciationAsPercentageOfRevenue: number;
  changeInWorkingCapitalAsPercentageOfRevenue: number;
  terminalGrowthRate: number;
  discountRate: number;
}

export interface ProjectedFinancialRow {
  year: number;
  revenue: number;
  ebitda: number;
  depreciation: number;
  ebit: number;
  taxes: number;
  nopat: number;
  capex: number;
  changeInNwc: number;
  unleveredFreeCashFlow: number;
}

export interface DCFRow {
    year: number;
    unleveredFreeCashFlow: number;
    discountFactor: number;
    presentValue: number;
}

export interface DcfValuationResult {
  intrinsicValue: number;
  terminalValue: number;
  enterpriseValue: number;
  equityValue: number;
  impliedSharePrice: number;
  projectedFinancials: ProjectedFinancialRow[];
  dcfAnalysis: DCFRow[];
}

export interface RelativeValuationResult {
    impliedSharePrice: number;
    commentary: string;
    comparableCompanies: { name: string; ticker: string; peRatio: number }[];
}

export interface DdmResult {
    impliedSharePrice: number;
    commentary: string;
}

export interface AssetBasedResult {
    impliedSharePrice: number;
    commentary: string;
}

export interface MultiModelValuationResult {
    dcf: DcfValuationResult;
    relative: RelativeValuationResult;
    ddm: DdmResult;
    assetBased: AssetBasedResult;
    commentary: string;
    currentSharePrice?: number;
    netDebt?: number;
    sharesOutstanding?: number;
}