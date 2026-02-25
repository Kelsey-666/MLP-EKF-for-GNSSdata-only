function strOut = cleanStruct(strIn)
% This function removes "useless" fields from an input structure, i.e.
% removes numeric arrays filled with all NaN and empty fields.
% input strIn = input structure
% output strOut
% this is a recursive function to deal with sub-structs
strOut = strIn;
fields = fieldnames(strOut);
for k = 1:length(fields)
    x = strOut.(fields{k});
    rmv = false;
    if(isempty(x))
        rmv = true;
    elseif(isnumeric(x))
        rmv = all(isnan(x(:)));
    elseif(isstruct(x))
        x = cleanStruct(x);
        if(isempty(x))
            rmv = true;
        else
            strOut.(fields{k}) = x;
        end
    end
    if(rmv)
        strOut = rmfield(strOut,char(fields(k)));
    end
end
end