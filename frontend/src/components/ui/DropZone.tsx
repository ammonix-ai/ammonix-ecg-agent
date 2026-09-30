import { cn } from '@/design/cn';
import { t } from '@/i18n/t';
import { Upload } from 'lucide-react';
import { useRef, useState, useCallback, type DragEvent } from 'react';

interface DropZoneProps {
  accept?: string;
  onDrop: (files: File[]) => void;
  multiple?: boolean;
  className?: string;
}

export function DropZone({ accept, onDrop, multiple = false, className }: DropZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragOver(false);
      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) {
        onDrop(multiple ? files : files.slice(0, 1));
      }
    },
    [onDrop, multiple],
  );

  const handleFileSelect = useCallback(() => {
    const files = inputRef.current?.files;
    if (files && files.length > 0) {
      onDrop(Array.from(files));
      if (inputRef.current) inputRef.current.value = '';
    }
  }, [onDrop]);

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        'flex flex-col items-center justify-center gap-3 p-8',
        'border-2 border-dashed rounded-[var(--radius-card)] transition-colors cursor-pointer',
        isDragOver
          ? 'border-brand bg-brand-50'
          : 'border-[var(--border-default)] hover:border-brand-300',
        className,
      )}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click();
      }}
    >
      <Upload className={cn('w-8 h-8', isDragOver ? 'text-brand' : 'text-text-muted')} />
      <div className="text-center">
        <p className="text-body text-text-primary">
          {t('dropzone.label', 'Drag & drop files here')}
        </p>
        <p className="text-caption text-text-muted mt-1">
          {t('dropzone.browse', 'or')}{' '}
          <span className="text-brand font-medium underline">
            {t('dropzone.browseLink', 'browse')}
          </span>
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={handleFileSelect}
        className="hidden"
        tabIndex={-1}
      />
    </div>
  );
}
