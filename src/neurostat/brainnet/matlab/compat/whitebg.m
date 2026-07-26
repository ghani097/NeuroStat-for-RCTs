function whitebg(fig, color)
%WHITEBG Minimal compatibility shim required by older BrainNet releases.
if nargin < 1 || isempty(fig)
    fig = gcf;
end
if nargin < 2 || isempty(color)
    color = [1 1 1];
end
set(fig, 'Color', color);
axes_handles = findall(fig, 'Type', 'axes');
for k = 1:numel(axes_handles)
    set(axes_handles(k), 'Color', color);
end
end
