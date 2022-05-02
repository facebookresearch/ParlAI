/*********************************************
* @ Jessica Huynh, Ting-Rui Chiang, Kyusong Lee 
* Carnegie Mellon University 2022
*********************************************/

export default function sessCateReducer(state, action) {
    switch(action.type) {
      case 'SESSION_CATEGORY_UPDATE':
        return action.category;
      default:
        return state || []
    }
  }
  
  