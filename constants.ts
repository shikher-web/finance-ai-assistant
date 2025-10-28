import type { NavItem } from './types';
import { ChartBarIcon, NewspaperIcon, CalculatorIcon, DocumentReportIcon } from './components/icons';

export const NAV_ITEMS: NavItem[] = [
  {
    id: 'analysis',
    label: 'Company Analysis',
    icon: ChartBarIcon,
  },
  {
    id: 'news',
    label: 'Market News',
    icon: NewspaperIcon,
  },
  {
    id: 'valuation',
    label: 'Valuation Models',
    icon: CalculatorIcon,
  },
  {
    id: 'reports',
    label: 'Generate Report',
    icon: DocumentReportIcon,
  },
];