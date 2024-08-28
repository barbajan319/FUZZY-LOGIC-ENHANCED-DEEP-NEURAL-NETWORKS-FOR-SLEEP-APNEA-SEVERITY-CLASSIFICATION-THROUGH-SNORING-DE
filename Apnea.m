data = readtable("severity_data.csv");

input = table2array(data(:,1:3));


Model = readfis  ('Apnea Severity');

Y=evalfis(input, Model);
Severity = (evalfis(input, Model));
rules = showrule(Model);
showrule(Model, Name, value)
