/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

function getStyle(stateStyle, defaultStyles={}) {
  let styles = {
    global: {
      spacing: 0
    },
    background: {
      "marginTop": "-3px",
      "marginBottom": "-3px",
      "fontSize": 15
    },
    instruction: {
      fontSize: 18,
    },
    example: {
      fontSize: 14,
      textAlign: "left",
      fontFamily: "Arial"
    },
    context: {
      fontSize: 14,
      lineHeight: 1.5
    },
    response: {
      fontSize: 14
    },
    question: {
      fontSize: 14
    },
    tabTitle: {},
    ...defaultStyles
  };
  for (const [section, style] of Object.entries(stateStyle || {})) {
    for (const [key, val] of Object.entries(style)) {
      styles[section] = styles[section] || {};
      if (['fontSize', "spacing"].indexOf(key) !== -1 && val !== undefined) {
        styles[section][key] = parseInt(val);
      } else {
        styles[section][key] = val;
      }
    }
  }
  styles.instruction.marginTop = `${styles.global.spacing}px`;
  styles.background.marginTop = `${styles.global.spacing}px`;
  styles.background.marginBottom = `${styles.global.spacing}px`;
  styles.context.marginBottom = `${Math.max(styles.context.lineHeight * 0.33, 0)}em`;

  return styles;
}

export {getStyle};
