import type { MermaidConfig } from "mermaid";

/**
 * Generates a Mermaid theme configuration that matches your design system.
 * Maps CSS variables to Mermaid's theming system for consistent look & feel.
 */
export function getMermaidConfig(isDarkTheme: boolean): MermaidConfig {
  // Colors from your design system (global.css)
  const colors = isDarkTheme
    ? {
        // Dark theme colors
        background: "#121926", // --background: 18 25 38
        foreground: "#f8fafc", // --foreground: 248 250 252
        card: "#121926", // --card: 18 25 38
        border: "#364152", // --gray-700: brighter border for visibility
        muted: "#202939", // --muted: 32 41 57
        mutedForeground: "#9aa4b2", // --muted-foreground: 154 164 178
        primary: "#102a56", // --primary: 16 42 86
        primaryForeground: "#fcfcfd", // --primary-foreground: 252 252 253
        secondary: "#0d121c", // --secondary: 13 18 28
        accent: "#202939", // --accent: 32 41 57
        link: "#2e90fa", // --link: 46 144 250
        // Use a brighter color for lines/edges
        lineColor: "#9aa4b2", // Use muted-foreground for better visibility
        success: "#17b26a", // --detail-success: 23 178 106
        failure: "#f04438", // --detail-failure: 240 68 56
        warning: "#fdb022", // --detail-warning: 253 176 34
      }
    : {
        // Light theme colors
        background: "#f8fafc", // --background: 248 250 252 (using card color for better contrast)
        foreground: "#0d121c", // --foreground: 13 18 28
        card: "#f8fafc", // --card: 248 250 252
        border: "#9aa4b2", // --gray-400: darker border for visibility
        muted: "#e3e8ef", // --muted: 227 232 239
        mutedForeground: "#697586", // --muted-foreground: 105 117 134
        primary: "#175cd3", // --primary: 23 92 211
        primaryForeground: "#fcfcfd", // --primary-foreground: 252 252 253
        secondary: "#ffffff", // --secondary: 255 255 255
        accent: "#e3e8ef", // --accent: 227 232 239
        link: "#2e90fa", // --link: 46 144 250
        // Use a darker color for lines/edges
        lineColor: "#697586", // Use muted-foreground for better visibility
        success: "#17b26a", // --detail-success: 23 178 106
        failure: "#f04438", // --detail-failure: 240 68 56
        warning: "#fdb022", // --detail-warning: 253 176 34
      };

  return {
    startOnLoad: false,
    securityLevel: "loose",
    theme: "base",
    // Ensure the SVG has proper sizing for text
    deterministicIds: true,
    themeVariables: {
      // General
      fontFamily: "Inter, sans-serif",
      fontSize: "14px",

      // Background colors
      background: colors.background,
      mainBkg: colors.card,
      secondBkg: colors.muted,
      tertiaryColor: colors.accent,

      // Text colors
      primaryTextColor: colors.foreground,
      secondaryTextColor: colors.mutedForeground,
      tertiaryTextColor: colors.foreground,
      textColor: colors.foreground,

      // Primary colors (nodes, shapes)
      primaryColor: colors.primary,
      primaryBorderColor: colors.link,

      // Secondary colors
      secondaryColor: colors.muted,
      secondaryBorderColor: colors.border,

      // Lines and borders - use brighter lineColor for visibility
      lineColor: colors.lineColor,
      border1: colors.border,
      border2: colors.border,

      // Note styling
      noteBkgColor: colors.muted,
      noteTextColor: colors.foreground,
      noteBorderColor: colors.border,

      // Actor styling (sequence diagrams)
      actorBkg: colors.card,
      actorBorder: colors.border,
      actorTextColor: colors.foreground,
      actorLineColor: colors.border,

      // Signal/message styling
      signalColor: colors.foreground,
      signalTextColor: colors.foreground,

      // Label styling
      labelBoxBkgColor: colors.card,
      labelBoxBorderColor: colors.border,
      labelTextColor: colors.foreground,

      // Loop styling
      loopTextColor: colors.foreground,

      // Activation styling (sequence diagrams)
      activationBkgColor: colors.muted,
      activationBorderColor: colors.border,

      // Sequence numbers
      sequenceNumberColor: colors.primaryForeground,

      // Edge labels
      edgeLabelBackground: colors.card,

      // Flowchart specific
      nodeBorder: colors.border,
      clusterBkg: colors.muted,
      clusterBorder: colors.border,
      defaultLinkColor: colors.link,

      // State diagram
      labelColor: colors.foreground,
      altBackground: colors.muted,

      // Class diagram
      classText: colors.foreground,

      // Git graph
      git0: colors.link,
      git1: colors.success,
      git2: colors.warning,
      git3: colors.failure,
      git4: colors.primary,
      git5: colors.muted,
      git6: colors.accent,
      git7: colors.secondary,
      gitBranchLabel0: colors.primaryForeground,
      gitBranchLabel1: colors.primaryForeground,
      gitBranchLabel2: colors.foreground,
      gitBranchLabel3: colors.primaryForeground,
      commitLabelColor: colors.foreground,
      commitLabelBackground: colors.card,

      // Pie chart
      pie1: colors.link,
      pie2: colors.success,
      pie3: colors.warning,
      pie4: colors.failure,
      pie5: colors.primary,
      pie6: colors.muted,
      pie7: colors.accent,
      pieStrokeColor: colors.border,
      pieStrokeWidth: "1px",
      pieTitleTextSize: "16px",
      pieTitleTextColor: colors.foreground,
      pieSectionTextSize: "12px",
      pieSectionTextColor: colors.foreground,
      pieLegendTextSize: "12px",
      pieLegendTextColor: colors.foreground,
      pieOpacity: "0.9",

      // Gantt chart
      taskBkgColor: colors.primary,
      taskBorderColor: colors.link,
      taskTextColor: colors.primaryForeground,
      taskTextLightColor: colors.primaryForeground,
      taskTextDarkColor: colors.foreground,
      activeTaskBkgColor: colors.link,
      activeTaskBorderColor: colors.primary,
      doneTaskBkgColor: colors.muted,
      doneTaskBorderColor: colors.border,
      critBkgColor: colors.failure,
      critBorderColor: colors.failure,
      todayLineColor: colors.warning,
      gridColor: colors.border,
      sectionBkgColor: colors.muted,
      sectionBkgColor2: colors.accent,
      altSectionBkgColor: colors.card,

      // Requirement diagram
      requirementBackground: colors.card,
      requirementBorderColor: colors.border,
      requirementBorderSize: "1px",
      requirementTextColor: colors.foreground,
      relationColor: colors.link,
      relationLabelBackground: colors.card,
      relationLabelColor: colors.foreground,
    },
    flowchart: {
      htmlLabels: false,
      curve: "basis",
      padding: 25,
      nodeSpacing: 50,
      rankSpacing: 60,
      useMaxWidth: true,
    },
    sequence: {
      useMaxWidth: true,
      boxMargin: 10,
      boxTextMargin: 5,
      noteMargin: 10,
      messageMargin: 35,
      mirrorActors: false,
      actorFontSize: 14,
      actorFontFamily: "Inter, sans-serif",
      noteFontSize: 13,
      noteFontFamily: "Inter, sans-serif",
      messageFontSize: 14,
      messageFontFamily: "Inter, sans-serif",
    },
    gantt: {
      useMaxWidth: true,
      barHeight: 20,
      barGap: 4,
      topPadding: 50,
      leftPadding: 75,
      gridLineStartPadding: 35,
      fontSize: 12,
      numberSectionStyles: 4,
    },
    pie: {
      useMaxWidth: true,
      textPosition: 0.75,
    },
    mindmap: {
      useMaxWidth: true,
      padding: 10,
    },
    gitGraph: {
      useMaxWidth: true,
    },
  };
}
