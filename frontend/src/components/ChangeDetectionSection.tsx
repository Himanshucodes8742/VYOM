import { useState } from 'react';
import {
  Activity,
  Sliders,
  Layers,
  Download,
  AlertTriangle,
  CheckCircle2,
  Maximize2,
  ChevronDown,
  ChevronUp,
  MapPin,
  RefreshCw,
  Sparkles,
  FileSpreadsheet
} from 'lucide-react';
import { ChangeDetectionResult, ChangeRegion } from '../types';
import ReticleCorners from './ReticleCorners';

interface ChangeDetectionSectionProps {
  changeResult: ChangeDetectionResult | null;
  isDetecting: boolean;
  onDetectChanges: (threshold: number, minArea: number) => void;
  showToast: (msg: string, type: 'info' | 'success') => void;
}

export default function ChangeDetectionSection({
  changeResult,
  isDetecting,
  onDetectChanges,
  showToast,
}: ChangeDetectionSectionProps) {
  const [activeLayer, setActiveLayer] = useState<'overlay' | 'heatmap' | 'mask' | 'registered'>('overlay');
  const [threshold, setThreshold] = useState<number>(30);
  const [minArea, setMinArea] = useState<number>(50);
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [selectedRegionId, setSelectedRegionId] = useState<number | null>(null);

  const handleDownload = (dataUrl: string | null, filename: string) => {
    if (!dataUrl) {
      showToast('Image not available for download.', 'info');
      return;
    }
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = filename;
    a.click();
    showToast(`Downloaded ${filename}`, 'success');
  };

  const handleExportCsv = (regions: ChangeRegion[]) => {
    if (!regions || regions.length === 0) {
      showToast('No detected regions to export.', 'info');
      return;
    }
    const header = 'id,area_px2,centroid_x,centroid_y,bbox_x,bbox_y,bbox_w,bbox_h,mean_delta_intensity\n';
    const rows = regions
      .map(
        (r) =>
          `${r.id},${r.area},${r.centroid[0]},${r.centroid[1]},${r.bbox[0]},${r.bbox[1]},${r.bbox[2]},${r.bbox[3]},${r.mean_intensity_diff}`
      )
      .join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lunar_surface_changes.csv';
    a.click();
    URL.revokeObjectURL(url);
    showToast(`Exported ${regions.length} change regions to CSV`, 'success');
  };

  const getCurrentImage = () => {
    if (!changeResult) return null;
    switch (activeLayer) {
      case 'overlay':
        return changeResult.change_overlay || changeResult.change_mask;
      case 'heatmap':
        return changeResult.diff_heatmap;
      case 'mask':
        return changeResult.change_mask;
      case 'registered':
        return changeResult.registered_image;
      default:
        return changeResult.change_overlay;
    }
  };

  return (
    <div className="bg-[#0A0E14] border border-[#1E293B] rounded-xl p-4 sm:p-5 shadow-2xl space-y-4 animate-fadeIn">
      {/* SECTION HEADER */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1A2536] pb-3.5">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.2)]">
            <Activity className="w-4 h-4 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-mono text-xs sm:text-sm font-bold text-slate-100 tracking-wider uppercase">
                LUNAR SURFACE CHANGE DETECTION
              </h3>
              <span className="px-1.5 py-0.5 rounded bg-emerald-950/80 border border-emerald-700/60 font-mono text-[9px] text-emerald-400 font-bold">
                MODULE 6 ACTIVE
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-sans">
              Illumination-normalized pixel difference &amp; morphological anomaly segmentation
            </p>
          </div>
        </div>

        {/* CONTROLS TOGGLE & RUN BUTTON */}
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`px-2.5 py-1.5 rounded font-mono text-xs border flex items-center gap-1.5 transition-all cursor-pointer ${
              showSettings
                ? 'bg-[#152336] text-[#3FD0E0] border-[#3FD0E0]/50'
                : 'bg-[#0E1522] text-slate-400 border-slate-700 hover:text-slate-200'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Parameters</span>
            {showSettings ? <ChevronUp className="w-3 h-3 ml-0.5" /> : <ChevronDown className="w-3 h-3 ml-0.5" />}
          </button>

          <button
            onClick={() => onDetectChanges(threshold, minArea)}
            disabled={isDetecting}
            className="px-3.5 py-1.5 bg-gradient-to-r from-emerald-500 to-teal-500 text-black font-mono text-xs font-bold rounded shadow-[0_0_16px_rgba(16,185,129,0.3)] hover:brightness-110 active:scale-98 transition-all flex items-center gap-2 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isDetecting ? 'animate-spin' : ''}`} />
            <span>{isDetecting ? 'ANALYZING SURFACE...' : 'DETECT CHANGES'}</span>
          </button>
        </div>
      </div>

      {/* PARAMETERS CONFIG PANEL */}
      {showSettings && (
        <div className="bg-[#0E1624] border border-[#1C2C40] rounded-lg p-3.5 grid grid-cols-1 sm:grid-cols-2 gap-4 animate-slideDown">
          <div className="space-y-1.5">
            <div className="flex justify-between items-center font-mono text-xs">
              <span className="text-slate-300 font-bold">Intensity Threshold:</span>
              <span className="text-[#3FD0E0] font-bold">{threshold} Δ</span>
            </div>
            <input
              type="range"
              min={15}
              max={80}
              step={1}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-full accent-cyan-400 bg-slate-800 rounded cursor-pointer"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-500">
              <span>15 (Sensitive)</span>
              <span>Default: 30</span>
              <span>80 (High-Contrast)</span>
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between items-center font-mono text-xs">
              <span className="text-slate-300 font-bold">Min Region Area:</span>
              <span className="text-amber-400 font-bold">{minArea} px²</span>
            </div>
            <input
              type="range"
              min={10}
              max={300}
              step={5}
              value={minArea}
              onChange={(e) => setMinArea(Number(e.target.value))}
              className="w-full accent-amber-400 bg-slate-800 rounded cursor-pointer"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-500">
              <span>10 px² (Tiny Boulders)</span>
              <span>Default: 50 px²</span>
              <span>300 px² (Large Craters)</span>
            </div>
          </div>
        </div>
      )}

      {/* LOADING STATE */}
      {isDetecting && (
        <div className="py-12 flex flex-col items-center justify-center space-y-3 bg-[#070B10]/80 rounded-lg border border-slate-800 animate-pulse">
          <div className="w-10 h-10 border-2 border-emerald-500/20 border-t-emerald-400 rounded-full animate-spin" />
          <div className="font-mono text-xs text-emerald-400 font-bold tracking-wider">
            COMPUTING ILLUMINATION-NORMALIZED RESIDUALS...
          </div>
          <div className="text-[11px] text-slate-500 font-mono">
            Applying CLAHE normalization &bull; Morphological opening &bull; Connected components
          </div>
        </div>
      )}

      {/* RESULTS PRESENTATION */}
      {!isDetecting && changeResult && changeResult.success && (
        <div className="space-y-4">
          {/* SUMMARY TELEMETRY CARDS */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {/* Card 1: Change Percentage */}
            <div className="bg-[#0E1522] border border-[#1E2B3D] rounded-lg p-3 relative overflow-hidden">
              <div className="font-mono text-[10px] text-slate-400 font-bold uppercase tracking-wider">
                SURFACE CHANGE AREA
              </div>
              <div className="mt-1 flex items-baseline gap-1.5">
                <span
                  className={`font-mono text-2xl font-black ${
                    changeResult.change_percentage > 2.0
                      ? 'text-red-400'
                      : changeResult.change_percentage > 0.5
                      ? 'text-amber-400'
                      : 'text-emerald-400'
                  }`}
                >
                  {changeResult.change_percentage.toFixed(2)}%
                </span>
                <span className="text-[10px] font-mono text-slate-500">OF OVERLAP</span>
              </div>
              <div className="mt-1 flex items-center gap-1 text-[10px] font-mono text-slate-400">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    changeResult.change_percentage > 2.0
                      ? 'bg-red-400'
                      : changeResult.change_percentage > 0.5
                      ? 'bg-amber-400'
                      : 'bg-emerald-400'
                  }`}
                />
                <span>
                  {changeResult.change_percentage > 2.0
                    ? 'SIGNIFICANT ANOMALY'
                    : changeResult.change_percentage > 0.5
                    ? 'MODERATE SHIFT'
                    : 'SUBTLE / LOCALIZED'}
                </span>
              </div>
            </div>

            {/* Card 2: Changed Pixels */}
            <div className="bg-[#0E1522] border border-[#1E2B3D] rounded-lg p-3">
              <div className="font-mono text-[10px] text-slate-400 font-bold uppercase tracking-wider">
                CHANGED PIXELS
              </div>
              <div className="mt-1 flex items-baseline gap-1.5">
                <span className="font-mono text-2xl font-black text-[#3FD0E0]">
                  {changeResult.changed_pixels.toLocaleString()}
                </span>
                <span className="text-[10px] font-mono text-slate-500">px</span>
              </div>
              <div className="text-[10px] font-mono text-slate-500 mt-1">
                Out of {changeResult.total_valid_pixels.toLocaleString()} valid px
              </div>
            </div>

            {/* Card 3: Detected Regions Count */}
            <div className="bg-[#0E1522] border border-[#1E2B3D] rounded-lg p-3">
              <div className="font-mono text-[10px] text-slate-400 font-bold uppercase tracking-wider">
                DETECTED REGIONS
              </div>
              <div className="mt-1 flex items-baseline gap-1.5">
                <span className="font-mono text-2xl font-black text-amber-400">
                  {changeResult.region_count}
                </span>
                <span className="text-[10px] font-mono text-slate-500">CLUSTERS</span>
              </div>
              <div className="text-[10px] font-mono text-slate-500 mt-1">
                Filter: &gt;= {changeResult.min_region_area} px²
              </div>
            </div>

            {/* Card 4: Largest Anomaly */}
            <div className="bg-[#0E1522] border border-[#1E2B3D] rounded-lg p-3">
              <div className="font-mono text-[10px] text-slate-400 font-bold uppercase tracking-wider">
                LARGEST REGION
              </div>
              <div className="mt-1 flex items-baseline gap-1.5">
                <span className="font-mono text-2xl font-black text-purple-300">
                  {changeResult.regions.length > 0 ? changeResult.regions[0].area.toLocaleString() : 0}
                </span>
                <span className="text-[10px] font-mono text-slate-500">px²</span>
              </div>
              <div className="text-[10px] font-mono text-slate-500 mt-1">
                Peak Δ: {changeResult.regions.length > 0 ? changeResult.regions[0].mean_intensity_diff.toFixed(1) : 0}
              </div>
            </div>
          </div>

          {/* MAIN DUAL DISPLAY: INTERACTIVE VIEWPORT (8 COLS) + REGIONS TABLE (4 COLS) */}
          <div className="grid grid-cols-12 gap-4">
            {/* VIEWPORT CANVAS (COL 12 LG:8) */}
            <div className="col-span-12 lg:col-span-8 bg-[#070B10] border border-[#1A2638] rounded-xl p-3 flex flex-col justify-between shadow-xl">
              <div>
                {/* LAYER CONTROLS BAR */}
                <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5 pb-2 border-b border-[#141E2C]">
                  <div className="flex items-center gap-1 font-mono text-xs">
                    <Layers className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-slate-300 font-bold uppercase text-[10px]">LAYER:</span>
                    <button
                      onClick={() => setActiveLayer('overlay')}
                      className={`px-2 py-0.5 rounded text-[10px] transition-colors cursor-pointer ${
                        activeLayer === 'overlay'
                          ? 'bg-rose-500/20 text-rose-300 border border-rose-500/50 font-bold'
                          : 'text-slate-500 hover:text-slate-300'
                      }`}
                    >
                      Highlight Overlay
                    </button>
                    <button
                      onClick={() => setActiveLayer('heatmap')}
                      className={`px-2 py-0.5 rounded text-[10px] transition-colors cursor-pointer ${
                        activeLayer === 'heatmap'
                          ? 'bg-amber-500/20 text-amber-300 border border-amber-500/50 font-bold'
                          : 'text-slate-500 hover:text-slate-300'
                      }`}
                    >
                      Thermal Delta
                    </button>
                    <button
                      onClick={() => setActiveLayer('mask')}
                      className={`px-2 py-0.5 rounded text-[10px] transition-colors cursor-pointer ${
                        activeLayer === 'mask'
                          ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 font-bold'
                          : 'text-slate-500 hover:text-slate-300'
                      }`}
                    >
                      Binary Mask
                    </button>
                    <button
                      onClick={() => setActiveLayer('registered')}
                      className={`px-2 py-0.5 rounded text-[10px] transition-colors cursor-pointer ${
                        activeLayer === 'registered'
                          ? 'bg-purple-500/20 text-purple-300 border border-purple-500/50 font-bold'
                          : 'text-slate-500 hover:text-slate-300'
                      }`}
                    >
                      Registered Source
                    </button>
                  </div>

                  {/* DOWNLOAD VIEW BUTTON */}
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() =>
                        handleDownload(
                          getCurrentImage(),
                          `change_${activeLayer}.png`
                        )
                      }
                      className="px-2 py-0.5 text-[10px] font-mono text-slate-400 hover:text-emerald-400 border border-slate-700 hover:border-emerald-500/40 rounded flex items-center gap-1 transition-colors cursor-pointer"
                    >
                      <Download className="w-3 h-3" />
                      <span>Save Layer</span>
                    </button>
                  </div>
                </div>

                {/* VIEWPORT IMAGE CONTAINER */}
                <div className="relative min-h-[380px] max-h-[500px] bg-black rounded-lg border border-[#162232] overflow-hidden flex items-center justify-center">
                  <ReticleCorners color="#10B981" size="w-3 h-3" />
                  {getCurrentImage() ? (
                    <img
                      src={getCurrentImage()!}
                      alt={`Change detection layer: ${activeLayer}`}
                      className="w-full h-auto max-h-[500px] object-contain"
                    />
                  ) : (
                    <div className="text-slate-600 font-mono text-xs">No layer data available.</div>
                  )}

                  {/* ACTIVE LAYER BADGE */}
                  <div className="absolute bottom-2.5 left-2.5 bg-black/80 backdrop-blur px-2 py-1 rounded border border-slate-800 text-[10px] font-mono flex items-center gap-1.5">
                    <span
                      className={`w-2 h-2 rounded-full ${
                        activeLayer === 'overlay'
                          ? 'bg-rose-400'
                          : activeLayer === 'heatmap'
                          ? 'bg-amber-400'
                          : activeLayer === 'mask'
                          ? 'bg-cyan-400'
                          : 'bg-purple-400'
                      }`}
                    />
                    <span className="text-slate-200 font-bold uppercase">{activeLayer}</span>
                  </div>

                  {/* COLOR KEY HINT */}
                  {activeLayer === 'overlay' && (
                    <div className="absolute bottom-2.5 right-2.5 bg-black/85 backdrop-blur px-2.5 py-1 rounded border border-rose-500/40 text-[10px] font-mono text-rose-300">
                      CORAL: Surface Change &bull; CYAN BOX: Region BBox
                    </div>
                  )}
                  {activeLayer === 'heatmap' && (
                    <div className="absolute bottom-2.5 right-2.5 bg-black/85 backdrop-blur px-2.5 py-1 rounded border border-amber-500/40 text-[10px] font-mono text-amber-300">
                      INFERNO: Black (0 Δ) &rarr; Orange (Mid) &rarr; Yellow (High Δ)
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* DETECTED REGIONS TABLE (COL 12 LG:4) */}
            <div className="col-span-12 lg:col-span-4 bg-[#0E1522] border border-[#1A2638] rounded-xl p-3 flex flex-col justify-between shadow-xl">
              <div>
                <div className="flex items-center justify-between pb-2 mb-2 border-b border-[#182333]">
                  <div className="flex items-center gap-1.5 font-mono text-xs font-bold text-slate-200 uppercase">
                    <MapPin className="w-3.5 h-3.5 text-amber-400" />
                    <span>DETECTED ANOMALIES</span>
                  </div>
                  <span className="text-[10px] font-mono text-slate-500">
                    {changeResult.regions.length} REGIONS
                  </span>
                </div>

                {changeResult.regions.length === 0 ? (
                  <div className="py-12 text-center text-slate-500 font-mono text-xs space-y-1">
                    <CheckCircle2 className="w-6 h-6 text-emerald-500/50 mx-auto mb-2" />
                    <div>NO ANOMALIES EXCEED THRESHOLD</div>
                    <div className="text-[10px] text-slate-600">
                      Surface is stable or try lowering threshold in parameters.
                    </div>
                  </div>
                ) : (
                  <div className="max-h-[360px] overflow-y-auto space-y-2 pr-1 custom-scrollbar">
                    {changeResult.regions.map((reg) => (
                      <div
                        key={reg.id}
                        onClick={() => setSelectedRegionId(reg.id)}
                        className={`p-2.5 rounded-lg border transition-all cursor-pointer font-mono text-xs ${
                          selectedRegionId === reg.id
                            ? 'bg-[#182638] border-cyan-400/80 shadow-[0_0_12px_rgba(63,208,224,0.15)]'
                            : 'bg-[#0B111A] border-slate-800 hover:border-slate-700'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-bold text-amber-400">ANOMALY #{reg.id}</span>
                          <span className="text-[10px] text-slate-400 bg-slate-800/80 px-1.5 py-0.2 rounded">
                            {reg.area.toLocaleString()} px²
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-1 text-[10px] text-slate-400">
                          <div>
                            <span className="text-slate-500">Centroid:</span> ({reg.centroid[0]}, {reg.centroid[1]})
                          </div>
                          <div>
                            <span className="text-slate-500">Mean Δ:</span>{' '}
                            <span className="text-rose-300 font-bold">{reg.mean_intensity_diff}</span>
                          </div>
                          <div className="col-span-2 text-[9px] text-slate-500 truncate">
                            BBox: [x={reg.bbox[0]}, y={reg.bbox[1]}, w={reg.bbox[2]}, h={reg.bbox[3]}]
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* EXPORT CSV BUTTON */}
              <div className="mt-3 pt-2 border-t border-[#182333]">
                <button
                  onClick={() => handleExportCsv(changeResult.regions)}
                  className="w-full p-2 bg-[#3FD0E0]/10 border border-[#3FD0E0]/30 hover:bg-[#3FD0E0]/20 rounded font-mono text-[10px] font-bold text-[#3FD0E0] flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
                >
                  <FileSpreadsheet className="w-3.5 h-3.5" />
                  <span>EXPORT ANOMALIES (CSV)</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ERROR STATE */}
      {!isDetecting && changeResult && !changeResult.success && (
        <div className="p-4 rounded-lg bg-red-950/40 border border-red-500/40 text-red-200 font-mono text-xs space-y-1">
          <div className="flex items-center gap-2 font-bold text-red-400">
            <AlertTriangle className="w-4 h-4" />
            <span>CHANGE DETECTION ERROR</span>
          </div>
          <div className="text-[11px] text-slate-300 font-sans">{changeResult.error}</div>
        </div>
      )}
    </div>
  );
}
