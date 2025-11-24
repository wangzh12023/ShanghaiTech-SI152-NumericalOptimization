set CRUDE_OILS;     
set PRODUCTS;       

param Sulfur{CRUDE_OILS};
param Price{PRODUCTS};
param Cost{CRUDE_OILS};
param Demand{PRODUCTS};
param SulfurLimit{PRODUCTS};
param UB >= 0;

var x >= 0, <= Demand['x'];
var y >= 0, <= Demand['y'];
var A >= 0, <= UB;
var B >= 0, <= UB;
var C_x >= 0, <= UB;
var C_y >= 0, <= UB;
var P_x >= 0, <= UB;
var P_y >= 0, <= UB;

var p >= 0, <= 100;  

maximize Profit:
    Price['x'] * x + Price['y'] * y
  - Cost['A'] * A - Cost['B'] * B - Cost['C'] * (C_x + C_y);

subject to PoolFlowBalance:
    P_x + P_y = A + B;

subject to ProductXFlowBalance:
    x = P_x + C_x;

subject to ProductYFlowBalance:
    y = P_y + C_y;

subject to PoolSulfurBalance:
    Sulfur['A'] * A + Sulfur['B'] * B = p * (P_x + P_y);

subject to ProductXSulfurContent:
    p * P_x + Sulfur['C'] * C_x <= SulfurLimit['x'] * x;

subject to ProductYSulfurContent:
    p * P_y + Sulfur['C'] * C_y <= SulfurLimit['y'] * y;