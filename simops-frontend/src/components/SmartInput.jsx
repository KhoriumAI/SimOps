import React, { useState, useEffect, useRef } from 'react';
import { evaluate } from 'mathjs';
import { ChevronUp, ChevronDown, HelpCircle } from 'lucide-react';

/**
 * SmartInput
 * A production-grade engineering input component.
 * 
 * Features:
 * - Inline math evaluation (e.g. "50/2 + 10" -> 35)
 * - Unit awareness (display vs. raw value)
 * - Draggable label scrubbing
 * - Keyboard shortcuts (Up/Down for increment)
 * 
 * @param {string} label - Input label
 * @param {number} value - The numeric value (controlled state)
 * @param {function} onChange - Callback (value: number) => void
 * @param {string} units - Suffix to display (e.g. "mm", "W")
 * @param {number} step - Increment step (default 1)
 * @param {number} min - Min value
 * @param {number} max - Max value
 * @param {string} tooltip - Information tooltip text
 */
const SmartInput = ({
    label,
    value,
    onChange,
    units = "",
    step = 1,
    min = -Infinity,
    max = Infinity,
    tooltip = "",
    className = ""
}) => {
    // Internal string state for editing (allows "50/..." before enter)
    const [inputValue, setInputValue] = useState(value?.toString() || "");
    const [isFocused, setIsFocused] = useState(false);
    const inputRef = useRef(null);
    const draggingRef = useRef(false);
    const startXRef = useRef(0);
    const startValueRef = useRef(0);

    // Sync external value to internal string when not editing
    useEffect(() => {
        if (!isFocused) {
            setInputValue(formatValue(value));
        }
    }, [value, isFocused]);

    const formatValue = (val) => {
        if (val === null || val === undefined || isNaN(val)) return "";
        // Scientific notation for very small/large numbers
        if (Math.abs(val) > 0 && (Math.abs(val) < 1e-4 || Math.abs(val) >= 1e5)) {
            return val.toExponential(4);
        }
        // Round to reasonable precision to avoid floating point errors
        return parseFloat(val.toFixed(6)).toString();
    };

    const parseAndCommit = (str) => {
        try {
            // 1. Remove unit suffix if present to allow "50mm" -> "50"
            let cleanStr = str;
            if (units && str.endsWith(units)) {
                cleanStr = str.slice(0, -units.length);
            }

            // 2. Evaluate math expression
            const result = evaluate(cleanStr);

            if (typeof result === 'number' && !isNaN(result)) {
                // 3. Clamp
                const clamped = Math.min(Math.max(result, min), max);
                onChange(clamped);
                setInputValue(formatValue(clamped));
            } else {
                // Revert if invalid
                setInputValue(formatValue(value));
            }
        } catch (e) {
            // Revert on math error
            console.warn("SmartInput eval error:", e);
            setInputValue(formatValue(value));
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            inputRef.current.blur();
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            const inc = e.shiftKey ? step * 10 : (e.altKey ? step * 0.1 : step);
            const newValue = Math.min((value || 0) + inc, max);
            onChange(newValue);
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            const inc = e.shiftKey ? step * 10 : (e.altKey ? step * 0.1 : step);
            const newValue = Math.max((value || 0) - inc, min);
            onChange(newValue);
        }
    };

    const handleBlur = () => {
        setIsFocused(false);
        parseAndCommit(inputValue);
    };

    const handleFocus = () => {
        setIsFocused(true);
        // Select all on focus for quick overwrite
        inputRef.current.select();
    };

    // --- Scrubbing Logic ---
    const handleMouseDown = (e) => {
        if (!onChange) return;
        draggingRef.current = true;
        startXRef.current = e.clientX;
        startValueRef.current = value || 0;

        document.body.style.cursor = 'ew-resize';
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
    };

    const handleMouseMove = (e) => {
        if (!draggingRef.current) return;
        e.preventDefault();

        const deltaX = e.clientX - startXRef.current;
        // Sensitivity: Shift = fast, Alt = slow
        const multiplier = e.shiftKey ? 10 : (e.altKey ? 0.1 : 1);
        const deltaValue = deltaX * step * multiplier;

        let newValue = startValueRef.current + deltaValue;
        newValue = Math.min(Math.max(newValue, min), max);

        onChange(newValue);
    };

    const handleMouseUp = () => {
        draggingRef.current = false;
        document.body.style.cursor = '';
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
    };

    return (
        <div className={`flex flex-col group ${className}`}>
            <label
                className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-1 cursor-ew-resize select-none hover:text-primary transition-colors flex items-center justify-between group/label"
                onMouseDown={handleMouseDown}
            >
                <span className="flex items-center gap-1.5">
                    {label}
                    {tooltip && (
                        <div className="relative group/tip cursor-help">
                            <HelpCircle className="w-2.5 h-2.5 opacity-40 group-hover/label:opacity-80 transition-opacity" />
                            <div className="absolute left-full ml-2 top-0 px-2 py-1 bg-popover text-popover-foreground text-[10px] rounded border border-border shadow-xl opacity-0 group-hover/tip:opacity-100 pointer-events-none transition-opacity whitespace-normal w-48 z-50">
                                {tooltip}
                            </div>
                        </div>
                    )}
                </span>
                {units && <span className="opacity-50 ml-1">({units})</span>}
            </label>

            <div className="relative relative-input-container">
                <input
                    ref={inputRef}
                    type="text"
                    value={isFocused ? inputValue : `${inputValue} ${units}`.trim()}
                    onChange={(e) => setInputValue(e.target.value)}
                    onBlur={handleBlur}
                    onFocus={handleFocus}
                    onKeyDown={handleKeyDown}
                    className="w-full bg-input/50 border border-border rounded px-2 py-1 text-xs font-mono text-foreground focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all tabular-nums"
                />

                {/* Micro Steppers (Optional, visible on hover) */}
                <div className="absolute right-0.5 top-0.5 bottom-0.5 flex flex-col opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                        className="h-1/2 px-1 text-muted-foreground hover:text-foreground hover:bg-muted/50 rounded-sm flex items-center"
                        onClick={() => onChange(Math.min((value || 0) + step, max))}
                        tabIndex={-1}
                    >
                        <ChevronUp className="w-2 h-2" />
                    </button>
                    <button
                        className="h-1/2 px-1 text-muted-foreground hover:text-foreground hover:bg-muted/50 rounded-sm flex items-center"
                        onClick={() => onChange(Math.max((value || 0) - step, min))}
                        tabIndex={-1}
                    >
                        <ChevronDown className="w-2 h-2" />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SmartInput;
