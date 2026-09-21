import { useCallback, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { datasetsApi, type Dataset } from '../api/datasets'
import { Upload, Database, CheckCircle, AlertCircle, FileText, Trash2 } from 'lucide-react'
import clsx from 'clsx'

// ─── Upload Zone ──────────────────────────────────────────────────────────────
function UploadZone({ onUpload }: { onUpload: (file: File, name: string) => void }) {
  const [dragging, setDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [datasetName, setDatasetName] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file?.name.endsWith('.csv')) {
      setSelectedFile(file)
      setDatasetName(file.name.replace('.csv', ''))
    }
  }, [])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setDatasetName(file.name.replace('.csv', ''))
    }
  }

  const handleSubmit = () => {
    if (selectedFile && datasetName.trim()) {
      onUpload(selectedFile, datasetName.trim())
      setSelectedFile(null)
      setDatasetName('')
    }
  }

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={clsx(
          'border-2 border-dashed rounded-xl p-12 flex flex-col items-center cursor-pointer transition-all duration-300',
          dragging
            ? 'border-brand-400 bg-brand-500/10 scale-[1.01]'
            : 'border-surface-border hover:border-brand-500/50 hover:bg-brand-500/5',
        )}
      >
        <input ref={inputRef} type="file" accept=".csv" className="hidden" onChange={handleFileChange} />
        <div className={clsx(
          'w-16 h-16 rounded-2xl flex items-center justify-center mb-4 transition-all duration-300',
          selectedFile ? 'bg-drift-safe/20' : 'bg-brand-600/20',
        )}>
          {selectedFile
            ? <CheckCircle size={28} className="text-drift-safe" />
            : <Upload size={28} className="text-brand-400" />
          }
        </div>
        {selectedFile ? (
          <>
            <p className="text-slate-100 font-semibold">{selectedFile.name}</p>
            <p className="text-slate-500 text-sm mt-1">
              {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </>
        ) : (
          <>
            <p className="text-slate-300 font-medium">Drop your CSV file here</p>
            <p className="text-slate-500 text-sm mt-1">or click to browse</p>
            <p className="text-slate-700 text-xs mt-4 font-mono">Supports up to 500 MB</p>
          </>
        )}
      </div>

      {selectedFile && (
        <div className="flex gap-3 animate-slide-up">
          <input
            className="input flex-1"
            placeholder="Dataset name"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
          />
          <button onClick={handleSubmit} className="btn-primary px-6" disabled={!datasetName.trim()}>
            Upload
          </button>
          <button onClick={() => { setSelectedFile(null); setDatasetName('') }} className="btn-secondary px-4">
            <Trash2 size={14} />
          </button>
        </div>
      )}
    </div>
  )
}

// ─── Schema Preview ───────────────────────────────────────────────────────────
function SchemaPreview({ dataset }: { dataset: Dataset }) {
  const schema = dataset.schema_json || {}
  const cols = Object.entries(schema).slice(0, 12)

  return (
    <div className="overflow-x-auto rounded-lg border border-surface-border">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-surface-border bg-surface">
            <th className="px-3 py-2 text-left text-slate-500 font-mono">Column</th>
            <th className="px-3 py-2 text-left text-slate-500 font-mono">Type</th>
            <th className="px-3 py-2 text-right text-slate-500 font-mono">Nulls%</th>
            <th className="px-3 py-2 text-right text-slate-500 font-mono">Unique</th>
          </tr>
        </thead>
        <tbody>
          {cols.map(([col, info]) => (
            <tr key={col} className="border-b border-surface-border/50 hover:bg-surface-border/30 transition-colors">
              <td className="px-3 py-2 text-slate-200 font-mono">{col}</td>
              <td className="px-3 py-2">
                <span className={clsx(
                  'badge text-[10px]',
                  (info as { dtype: string }).dtype.includes('int') || (info as { dtype: string }).dtype.includes('float')
                    ? 'bg-brand-900/60 text-brand-300'
                    : 'bg-amber-900/40 text-amber-400',
                )}>
                  {(info as { dtype: string }).dtype}
                </span>
              </td>
              <td className="px-3 py-2 text-right text-slate-400">
                {(info as { null_pct: number }).null_pct.toFixed(1)}%
              </td>
              <td className="px-3 py-2 text-right text-slate-400 font-mono">
                {(info as { unique_count: number }).unique_count.toLocaleString()}
              </td>
            </tr>
          ))}
          {Object.keys(schema).length > 12 && (
            <tr>
              <td colSpan={4} className="px-3 py-2 text-slate-600 text-center text-[10px]">
                + {Object.keys(schema).length - 12} more columns
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}

// ─── Dataset Row ──────────────────────────────────────────────────────────────
function DatasetRow({ dataset, onSelect, selected }: {
  dataset: Dataset
  onSelect: (d: Dataset) => void
  selected: boolean
}) {
  const cols = dataset.schema_json ? Object.keys(dataset.schema_json).length : 0

  return (
    <div
      onClick={() => onSelect(dataset)}
      className={clsx(
        'flex items-center gap-4 px-4 py-3 rounded-lg cursor-pointer transition-all duration-200 border',
        selected
          ? 'bg-brand-600/10 border-brand-500/30'
          : 'border-transparent hover:bg-surface-border/40 hover:border-surface-border',
      )}
    >
      <div className="w-9 h-9 rounded-lg bg-brand-900/50 flex items-center justify-center flex-shrink-0">
        <Database size={16} className="text-brand-400" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-slate-200 truncate">{dataset.name}</p>
        <p className="text-xs text-slate-500 font-mono">
          {dataset.row_count ? `${parseInt(dataset.row_count).toLocaleString()} rows` : '—'} · {cols} cols
        </p>
      </div>
      <div className="text-right flex-shrink-0">
        <p className="text-[10px] text-slate-600 font-mono">
          {new Date(dataset.uploaded_at).toLocaleDateString()}
        </p>
      </div>
    </div>
  )
}

// ─── Main Page ────────────────────────────────────────────────────────────────
export default function DatasetUpload() {
  const queryClient = useQueryClient()
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [successMsg, setSuccessMsg] = useState<string | null>(null)

  const { data: datasets, isLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => datasetsApi.list(20, 0),
    refetchInterval: 5000,
  })

  const uploadMutation = useMutation({
    mutationFn: ({ file, name }: { file: File; name: string }) => datasetsApi.upload(file, name),
    onSuccess: (newDs) => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      setSuccessMsg(`"${newDs.name}" uploaded successfully — ${parseInt(newDs.row_count ?? '0').toLocaleString()} rows detected`)
      setSelectedDataset(newDs)
      setUploadError(null)
      setTimeout(() => setSuccessMsg(null), 5000)
    },
    onError: (err: unknown) => {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? 'Upload failed'
      setUploadError(msg)
    },
  })

  return (
    <div className="animate-slide-up max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="section-title">Dataset Upload</h1>
        <p className="text-sm text-slate-500 mt-1">
          Upload CSV datasets — schema is inferred and baseline stats computed for drift detection
        </p>
      </div>

      {/* Alerts */}
      {successMsg && (
        <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-drift-safe/10 border border-drift-safe/30 text-drift-safe text-sm animate-fade-in">
          <CheckCircle size={16} />
          {successMsg}
        </div>
      )}
      {uploadError && (
        <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-drift-critical/10 border border-drift-critical/30 text-drift-critical text-sm animate-fade-in">
          <AlertCircle size={16} />
          {uploadError}
        </div>
      )}

      <div className="grid grid-cols-5 gap-6">
        {/* Left — uploader + list */}
        <div className="col-span-2 space-y-4">
          <div className="card">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">New Dataset</p>
            <UploadZone onUpload={(file, name) => uploadMutation.mutate({ file, name })} />
            {uploadMutation.isPending && (
              <div className="mt-4 flex items-center gap-2 text-sm text-slate-400">
                <div className="w-4 h-4 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
                Uploading and computing baseline stats…
              </div>
            )}
          </div>

          <div className="card">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-3">
              All Datasets ({datasets?.total ?? 0})
            </p>
            {isLoading ? (
              <div className="text-center py-8 text-slate-600 text-sm">Loading…</div>
            ) : datasets?.items.length === 0 ? (
              <div className="text-center py-8 text-slate-700 text-sm">No datasets yet</div>
            ) : (
              <div className="space-y-1">
                {datasets?.items.map((d) => (
                  <DatasetRow
                    key={d.id}
                    dataset={d}
                    selected={selectedDataset?.id === d.id}
                    onSelect={setSelectedDataset}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right — schema preview */}
        <div className="col-span-3">
          {selectedDataset ? (
            <div className="card space-y-4 animate-fade-in">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Schema Preview</p>
                  <p className="text-lg font-bold text-slate-100 mt-1">{selectedDataset.name}</p>
                  <p className="text-xs text-slate-500 font-mono mt-0.5">
                    {selectedDataset.row_count
                      ? `${parseInt(selectedDataset.row_count).toLocaleString()} rows`
                      : '—'}{' '}
                    · {selectedDataset.schema_json ? Object.keys(selectedDataset.schema_json).length : 0} columns
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="badge badge-safe">Ready</span>
                </div>
              </div>
              {selectedDataset.schema_json && <SchemaPreview dataset={selectedDataset} />}

              {/* Dataset ID for copying */}
              <div className="pt-2 border-t border-surface-border">
                <p className="text-xs text-slate-600 font-mono">ID: {selectedDataset.id}</p>
              </div>
            </div>
          ) : (
            <div className="card h-full flex flex-col items-center justify-center py-20 text-center">
              <FileText size={32} className="text-slate-700 mb-3" />
              <p className="text-slate-500 text-sm">Select a dataset to preview its schema</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
