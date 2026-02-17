/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: "#667eea",
                "primary-light": "#7c3aed",
                secondary: "#06b6d4",
                accent: "#ec4899",
                success: "#10b981",
                warning: "#f59e0b",
                danger: "#ef4444",
                "bg-primary": "var(--bg-primary)",
                "bg-secondary": "var(--bg-secondary)",
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            },
            backgroundImage: {
                'gradient-main': 'linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 25%, rgba(236, 72, 153, 0.05) 50%, rgba(6, 182, 212, 0.05) 75%, rgba(102, 126, 234, 0.05) 100%)',
            }
        },
    },
    plugins: [require("daisyui")],
    daisyui: {
        themes: ["light", "dark"],
    },
}
