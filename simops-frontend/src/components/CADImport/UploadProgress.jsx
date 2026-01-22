import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

/**
 * Premium progress indicator with percentage and time estimation
 */
export default function UploadProgress({ progress, filename }) {
    const [estimatedSeconds, setEstimatedSeconds] = useState(0);

    useEffect(() => {
        if (progress > 0 && progress < 100) {
            // Very simple estimation logic for demo/UI feel
            const remaining = Math.max(0, (100 - progress) / 20);
            setEstimatedSeconds(remaining.toFixed(1));
        }
    }, [progress]);

    return (
        <div className="w-full space-y-3">
            <div className="flex justify-between items-center text-sm font-medium">
                <span className="text-gray-700 truncate max-w-[70%]">Uploading {filename}...</span>
                <span className="text-brand-600 tabular-nums">{Math.round(progress)}%</span>
            </div>

            <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden border border-gray-200/50">
                <motion.div
                    className="h-full bg-brand-500 shadow-[0_0_10px_rgba(var(--brand-rgb),0.3)]"
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                />
            </div>

            {progress < 100 && (
                <p className="text-[10px] text-gray-500 text-right italic">
                    Estimated {estimatedSeconds}s remaining...
                </p>
            )}
        </div>
    );
}
