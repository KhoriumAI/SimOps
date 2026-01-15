import { useState } from 'react'
import {
    Cpu, Wind, Settings, ChevronRight, Code, X, Check,
    Thermometer, Droplets, Flame
} from 'lucide-react'

/**
 * PhysicsTemplateSelector - "Physics Cards" Component
 * 
 * Displays simulation template options as selectable cards.
 * Featured templates are shown prominently, with an "Advanced" option
 * to view/edit raw JSON config.
 */
export default function PhysicsTemplateSelector({
    templates = [],
    selectedTemplate,
    onSelect,
    onConfigChange,
    featuredIds = ['electronics_cooling', 'heat_sink', 'led_housing']
}) {
    const [showAdvanced, setShowAdvanced] = useState(false)
    const [editedConfig, setEditedConfig] = useState('')
    const [configError, setConfigError] = useState(null)

    // Filter to featured templates
    const featuredTemplates = templates.filter(t => featuredIds.includes(t.id))
    const otherTemplates = templates.filter(t => !featuredIds.includes(t.id))

    // Get icon for template
    const getTemplateIcon = (id) => {
        switch (id) {
            case 'electronics_cooling':
            case 'pcb_analysis':
                return <Cpu className="w-6 h-6" />
            case 'heat_sink':
                return <Wind className="w-6 h-6" />
            case 'led_housing':
                return <Flame className="w-6 h-6" />
            case 'battery_pack':
                return <Thermometer className="w-6 h-6" />
            case 'cryogenic_vessel':
                return <Droplets className="w-6 h-6" />
            default:
                return <Settings className="w-6 h-6" />
        }
    }

    // Get gradient colors for template cards
    const getCardGradient = (id) => {
        switch (id) {
            case 'electronics_cooling':
            case 'pcb_analysis':
                return 'from-blue-500 to-blue-600'
            case 'heat_sink':
                return 'from-orange-500 to-orange-600'
            case 'led_housing':
                return 'from-amber-500 to-amber-600'
            case 'battery_pack':
                return 'from-green-500 to-green-600'
            case 'rocket_nozzle':
                return 'from-red-500 to-red-600'
            default:
                return 'from-gray-500 to-gray-600'
        }
    }

    const handleSelect = (template) => {
        onSelect?.(template)
        setEditedConfig(JSON.stringify(template.config, null, 2))
        setConfigError(null)
    }

    const handleAdvancedSave = () => {
        try {
            const parsed = JSON.parse(editedConfig)
            onConfigChange?.(parsed)
            setConfigError(null)
            setShowAdvanced(false)
        } catch (e) {
            setConfigError('Invalid JSON: ' + e.message)
        }
    }

    const openAdvanced = () => {
        if (selectedTemplate) {
            setEditedConfig(JSON.stringify(selectedTemplate.config, null, 2))
        }
        setShowAdvanced(true)
    }

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-gray-700">Physics Template</h3>
                {selectedTemplate && (
                    <button
                        onClick={openAdvanced}
                        className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-700"
                    >
                        <Code className="w-3 h-3" />
                        Advanced
                    </button>
                )}
            </div>

            {/* Featured Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {featuredTemplates.map((template) => (
                    <button
                        key={template.id}
                        onClick={() => handleSelect(template)}
                        className={`relative text-left rounded-lg overflow-hidden border-2 transition-all ${selectedTemplate?.id === template.id
                            ? 'border-blue-500 ring-2 ring-blue-200'
                            : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
                            }`}
                    >
                        {/* Card Header */}
                        <div className={`bg-gradient-to-r ${getCardGradient(template.id)} p-3 text-white`}>
                            <div className="flex items-center gap-2">
                                {getTemplateIcon(template.id)}
                                <span className="font-semibold text-sm truncate">{template.name}</span>
                            </div>
                        </div>

                        {/* Card Body */}
                        <div className="p-3 bg-white">
                            <p className="text-xs text-gray-600 line-clamp-2 mb-2">
                                {template.description}
                            </p>
                            <div className="flex items-center justify-between text-xs">
                                <span className="text-gray-500">{template.material}</span>
                                <ChevronRight className={`w-4 h-4 transition-colors ${selectedTemplate?.id === template.id ? 'text-blue-500' : 'text-gray-400'
                                    }`} />
                            </div>
                        </div>

                        {/* Selected indicator */}
                        {selectedTemplate?.id === template.id && (
                            <div className="absolute top-2 right-2 w-5 h-5 bg-white rounded-full flex items-center justify-center">
                                <Check className="w-3 h-3 text-blue-500" />
                            </div>
                        )}
                    </button>
                ))}
            </div>

            {/* Other Templates Dropdown */}
            {otherTemplates.length > 0 && (
                <details className="group">
                    <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700 list-none flex items-center gap-1">
                        <ChevronRight className="w-3 h-3 group-open:rotate-90 transition-transform" />
                        {otherTemplates.length} more templates
                    </summary>
                    <div className="mt-2 grid grid-cols-2 sm:grid-cols-4 gap-2">
                        {otherTemplates.map((template) => (
                            <button
                                key={template.id}
                                onClick={() => handleSelect(template)}
                                className={`text-left p-2 rounded border text-xs transition-all ${selectedTemplate?.id === template.id
                                    ? 'border-blue-500 bg-blue-50'
                                    : 'border-gray-200 hover:border-gray-300'
                                    }`}
                            >
                                <div className="font-medium text-gray-700 truncate">{template.name}</div>
                                <div className="text-gray-500 truncate">{template.material}</div>
                            </button>
                        ))}
                    </div>
                </details>
            )}

            {/* Template Inputs (when selected) */}
            {selectedTemplate && (
                <div className="space-y-3">
                    <div className="bg-gray-50 rounded-lg p-3 space-y-3">
                        <div className="text-xs font-medium text-gray-700 flex items-center gap-2">
                            <Thermometer className="w-3 h-3" />
                            Quick Settings
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                            {selectedTemplate.inputs?.slice(0, 4).map((input) => (
                                <div key={input.key}>
                                    <label className="block text-xs text-gray-500 mb-1">{input.label}</label>
                                    {input.type === 'select' ? (
                                        <select
                                            className="w-full text-xs border border-gray-300 rounded px-2 py-1.5 bg-white"
                                            defaultValue={input.default}
                                            onChange={(e) => onConfigChange?.({
                                                ...selectedTemplate.config,
                                                physics: { ...selectedTemplate.config.physics, [input.key]: e.target.value }
                                            })}
                                        >
                                            {input.options?.slice(0, 10).map(opt => (
                                                <option key={opt} value={opt}>{opt.replace(/_/g, ' ')}</option>
                                            ))}
                                        </select>
                                    ) : (
                                        <input
                                            type="number"
                                            className="w-full text-xs border border-gray-300 rounded px-2 py-1.5"
                                            defaultValue={input.default}
                                            onChange={(e) => onConfigChange?.({
                                                ...selectedTemplate.config,
                                                physics: { ...selectedTemplate.config.physics, [input.key]: parseFloat(e.target.value) }
                                            })}
                                        />
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Geometry & Contact Settings */}
                    <div className="bg-gray-50 rounded-lg p-3 space-y-2">
                        <div className="text-xs font-medium text-gray-700 flex items-center gap-2">
                            <Settings className="w-3 h-3" />
                            Geometry & Contact
                        </div>
                        <div className="flex flex-wrap items-center gap-4">
                            <label className="flex items-center gap-2 text-xs text-gray-600 cursor-pointer select-none">
                                <input
                                    type="checkbox"
                                    checked={selectedTemplate.config.contact?.use_tie_contacts || false}
                                    onChange={(e) => onConfigChange?.({
                                        ...selectedTemplate.config,
                                        contact: {
                                            ...selectedTemplate.config.contact,
                                            use_tie_contacts: e.target.checked
                                        }
                                    })}
                                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                />
                                Use TIE Constraints (Multi-volume)
                            </label>

                            {selectedTemplate.config.contact?.use_tie_contacts && (
                                <div className="flex items-center gap-2">
                                    <span className="text-xs text-gray-500">Tolerance:</span>
                                    <input
                                        type="number"
                                        step="0.001"
                                        className="w-20 text-xs border border-gray-300 rounded px-2 py-1"
                                        value={selectedTemplate.config.contact?.tie_tolerance || 0.001}
                                        onChange={(e) => onConfigChange?.({
                                            ...selectedTemplate.config,
                                            contact: {
                                                ...selectedTemplate.config.contact,
                                                tie_tolerance: parseFloat(e.target.value)
                                            }
                                        })}
                                    />
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Advanced Config Modal */}
            {showAdvanced && (
                <div
                    className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
                    onClick={() => setShowAdvanced(false)}
                >
                    <div
                        className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[80vh] overflow-hidden"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="px-4 py-3 bg-gray-800 text-white flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Code className="w-4 h-4" />
                                <span className="font-semibold">Advanced Configuration</span>
                            </div>
                            <button onClick={() => setShowAdvanced(false)} className="hover:bg-gray-700 p-1 rounded">
                                <X className="w-4 h-4" />
                            </button>
                        </div>

                        <div className="p-4">
                            <p className="text-xs text-gray-500 mb-2">
                                Edit the raw JSON configuration for full control over simulation parameters.
                            </p>
                            <textarea
                                value={editedConfig}
                                onChange={(e) => setEditedConfig(e.target.value)}
                                className="w-full h-80 font-mono text-xs p-3 border border-gray-300 rounded-lg bg-gray-50"
                                spellCheck={false}
                            />
                            {configError && (
                                <p className="text-xs text-red-600 mt-2">{configError}</p>
                            )}
                        </div>

                        <div className="px-4 py-3 bg-gray-50 border-t flex justify-end gap-2">
                            <button
                                onClick={() => setShowAdvanced(false)}
                                className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-200 rounded"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleAdvancedSave}
                                className="px-4 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                            >
                                Apply Changes
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
