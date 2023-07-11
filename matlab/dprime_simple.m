function [dp,c] = dprime_simple(h,fA)
% DPRIME_SIMPLE d' calculation given hit/false alarm rates
%   dp = dprime_simple(h,fA) returns the d' value dp 
%   for the hit rate h and false alarm rate fA
%   originally by Karin Cox 8/31/14
%   edits by Madeline Klinger Dec 2021  
%
%   inputs: 
%   h = hit rate, as float (0 < h < 1) 
%   fA = false alarm rate, as float (0 < fA < 1)
%
%   outputs: 
%   dp = d'
%   c = criterion c (negative values --> bias towards yes responses)
%
%   Example:
%   [dp,c] = dprime_simple(0.9,0.05)   
%   dp =
%     2.9264
%   c = 
%     0.1817
%
%   formulas: Stanislaw, H., & Todorov, N. (1999). Calculation of signal 
%   detection theory measures. Behavior research methods, instruments, & 
%   computers, 31(1), 137-149.


% check n args
narginchk(2,2);


% d prime = z(h)-z(fA)
% implement floor and ceiling


% check for values out of bounds, also issue errors or warnings if =1 or 0
if or(or(h>1,h<0),or(fA>1,fA<0))
    error('input arguments must fall in the 0 to 1 range')
% standard d' formula returns NaN or Inf results for 0 or 1 inputs,
% corrections may be required (see article above for suggestions)
elseif or(or(h==1,h==0),or(fA==1,fA==0))
    warning('This function will not return finite values when h or fA = 0 or 1.')
end


if h == 1 && fA == 0
    dp = norminv(0.99)-norminv(0.01); h = 0.99;
elseif h == 1 && fA == 1
    dp = norminv(0.99)-norminv(0.99);
elseif h == 0 && fA == 0
    dp = norminv(0.01)-norminv(0.01);
elseif h == 0 && fA == 1
    dp = norminv(0.01)-norminv(0.99);
elseif h == 1
    dp = norminv(0.99)-norminv(fA);
elseif h == 0
    dp = norminv(0.01)-norminv(fA);
elseif fA == 1
    dp = norminv(h)-norminv(0.99);
elseif fA == 0
    dp = norminv(h)-norminv(0.01);
else
    dp = norminv(h)-norminv(fA);
end

% c = -0.5*[z(h)+z(fA)]
c = -0.5*(norminv(h)+ norminv(fA));

end

