/** @type {import('tailwindcss').Config} */
module.exports = {
    darkMode: ["class"],
    // content: ["./src/**/*.{ts,tsx}"], // Adjust as needed for your project
    theme: {
        container: {
            center: true,
            padding: "2rem",
            screens: {
                "2xl": "1400px", // Widescreen support for data dashboards
            },
        },
        extend: {
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
                mono: ['JetBrains Mono', 'SF Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
            },
            fontSize: {
                // High-density scaling for technical data
                xs: ['0.75rem', { lineHeight: '1rem' }],
                sm: ['0.875rem', { lineHeight: '1.25rem' }],
                base: ['1rem', { lineHeight: '1.5rem' }],
            },
            boxShadow: {
                // The "Engineering" Shadow Scale
                // subtle depth for cards
                'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
                // default depth
                'DEFAULT': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
                // Recessed look for inputs (The "Slot" effect)
                'inner': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
                // Specialized ring for focus states
                'input': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
            },
            colors: {
                // Semantic aliases for dark mode support
                border: "hsl(var(--border))",
                input: "hsl(var(--input))",
                ring: "hsl(var(--ring))",
                background: "hsl(var(--background))",
                foreground: "hsl(var(--foreground))",
                primary: {
                    DEFAULT: "hsl(var(--primary))",
                    foreground: "hsl(var(--primary-foreground))",
                },
                secondary: {
                    DEFAULT: "hsl(var(--secondary))",
                    foreground: "hsl(var(--secondary-foreground))",
                },
                destructive: {
                    DEFAULT: "hsl(var(--destructive))",
                    foreground: "hsl(var(--destructive-foreground))",
                },
                muted: {
                    DEFAULT: "hsl(var(--muted))",
                    foreground: "hsl(var(--muted-foreground))",
                },
                accent: {
                    DEFAULT: "hsl(var(--accent))",
                    foreground: "hsl(var(--accent-foreground))",
                },
                popover: {
                    DEFAULT: "hsl(var(--popover))",
                    foreground: "hsl(var(--popover-foreground))",
                },
                card: {
                    DEFAULT: "hsl(var(--card))",
                    foreground: "hsl(var(--card-foreground))",
                },
            },
            borderRadius: {
                // Restricting "vibe" roundness
                lg: "var(--radius)",
                md: "calc(var(--radius) - 2px)",
                sm: "calc(var(--radius) - 4px)",
            },
        }
    }
}
