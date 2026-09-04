/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#07090d",
          900: "#0c1118",
          800: "#121923",
          700: "#1a2433",
          600: "#243044",
        },
        mint: {
          DEFAULT: "#3ee0c6",
          dim: "#1a6f64",
        },
        ember: "#ffb020",
        flare: "#ff4d6d",
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "ui-sans-serif", "system-ui"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 40px rgba(62, 224, 198, 0.12)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
