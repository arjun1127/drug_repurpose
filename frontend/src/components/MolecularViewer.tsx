import React, { useEffect, useRef, useState } from 'react';
import * as $3Dmol from '3dmol';

interface MolecularViewerProps {
  url: string;
  format: 'pdb' | 'sdf';
  title?: string;
  className?: string;
}

export const MolecularViewer: React.FC<MolecularViewerProps> = ({ 
  url, 
  format,
  title,
  className = ''
}) => {
  const viewerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let viewer: any = null;

    const initViewer = async () => {
      if (!viewerRef.current) return;
      
      try {
        setLoading(true);
        setError('');

        const response = await fetch(url);
        if (!response.ok) {
          throw new Error('Failed to load molecular data');
        }
        const data = await response.text();

        const config = { backgroundColor: '#1e293b' };
        viewer = $3Dmol.createViewer(viewerRef.current, config);

        viewer.addModel(data, format);

        if (format === 'pdb') {
          viewer.setStyle({}, { cartoon: { color: 'spectrum' } });
        } else {
          viewer.setStyle({}, { stick: { colorscheme: 'cyanCarbon' } });
        }

        viewer.zoomTo();
        viewer.render();
        setLoading(false);

      } catch (err: any) {
        console.error("3DMol Error:", err);
        setError(err.message || 'Error rendering molecule');
        setLoading(false);
      }
    };

    initViewer();

  }, [url, format]);

  return (
    <div className={`molecular-viewer-container ${className}`} style={{ width: '100%', height: '300px', position: 'relative', borderRadius: '8px', overflow: 'hidden', backgroundColor: '#1e293b', border: '1px solid rgba(255,255,255,0.1)' }}>
      {title && (
        <div style={{ position: 'absolute', top: 10, left: 10, zIndex: 10, color: 'white', backgroundColor: 'rgba(0,0,0,0.5)', padding: '4px 8px', borderRadius: '4px', fontSize: '12px' }}>
          {title}
        </div>
      )}
      
      {loading && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>
          Loading 3D Model...
        </div>
      )}
      
      {error && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)' }}>
          {error}
        </div>
      )}

      <div ref={viewerRef} style={{ width: '100%', height: '100%' }} className="viewer-canvas" />
    </div>
  );
};
