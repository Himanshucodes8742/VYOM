export type NavTab = 'register' | 'compare' | 'changes' | 'about';

export type SensorSource = 'OHRC' | 'TMC-2' | 'IIRS';
export type SensorReference = 'LROC_NAC' | 'LROC_WAC' | 'SELENE_TC';

export type AlgorithmName =
  | 'Adaptive (AKAZE -> RIFT2)'
  | 'RIFT2-style (Phase Congruency)'
  | 'Crater Landmarks (trained CNN)'
  | 'Learned Match Verifier (trained)'
  | 'Learned Descriptor (trained)'
  | 'SuperGlue (Deep Graph Neural Network)'
  | 'AKAZE (Non-linear Scale Space)'
  | 'SIFT (Scale-Invariant Feature Transform)';

export type PreprocessingName =
  | 'clahe'
  | 'photometric_clahe'
  | 'photometric'
  | 'histogram';

export interface Keypoint {
  id: string;
  x: number;
  y: number;
  refX: number;
  refY: number;
  dx: number;
  dy: number;
  residual: number;
  confidence: number;
}

export interface PipelineStep {
  name: string;
  sub: string;
  details: string;
}

export interface AlgorithmBenchmark {
  id: string;
  name: string;
  tag: string;
  engine: string;
  preprocessing?: string;
  rmse: number;
  inliers: number;
  ratio: number;
  score: number;
  runtime: number;
  isBestRmse?: boolean;
  isBestInliers?: boolean;
  isBestRatio?: boolean;
  isBestScore?: boolean;
  isBestRuntime?: boolean;
  description: string;
}

export interface SensorSpec {
  sensor: string;
  mission: string;
  resolution: string;
  band: string;
  swath: string;
}

export interface RegistrationMetrics {
  rmse: number;
  inlier_count: number;
  inlier_ratio: number;
  distribution_score: number;
  runtime?: number;
  inlierCount?: number;
  inlierRatio?: number;
  distributionScore?: number;
}

export interface DemoPair {
  id: string;
  title: string;
  source_sensor: string;
  reference_sensor: string;
  source_file: string | null;
  reference_file: string | null;
  description: string;
  files: string[];
  source_url: string | null;
  reference_url: string | null;
}

export interface ChangeRegion {
  id: number;
  label: number;
  area: number;
  bbox: [number, number, number, number];
  centroid: [number, number];
  mean_intensity_diff: number;
}

export interface ChangeDetectionResult {
  success: boolean;
  algorithm: string;
  preprocessing: string;
  sensor_pair: string;
  threshold: number;
  min_region_area: number;
  change_percentage: number;
  changed_pixels: number;
  total_valid_pixels: number;
  region_count: number;
  regions: ChangeRegion[];
  change_mask: string | null;
  change_overlay: string | null;
  diff_heatmap: string | null;
  registered_image: string | null;
  metrics?: RegistrationMetrics | null;
  transform_matrix?: number[][] | null;
  error?: string | null;
}

