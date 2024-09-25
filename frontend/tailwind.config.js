const colors = require("tailwindcss/colors");

module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: colors.indigo,
        secondary: colors.purple,
        neutral: colors.gray,
      },
    },
  },
  plugins: [
    require("@tailwindcss/forms"),
    // ... any other plugins you're using
  ],
};
