%NEUROSTAT_BRAINNET_ENTRY Strict JSON boundary for BrainNet Viewer rendering.
request_path = getenv('NEUROSTAT_BRAINNET_REQUEST');
assert(~isempty(request_path), 'NEUROSTAT_BRAINNET_REQUEST is missing.');
request = jsondecode(fileread(request_path));
assert(isfield(request, 'mode'), 'BrainNet request mode is missing.');
addpath(request.brainnet_root);
addpath(request.compat_dir, '-begin');
rehash;
expected_mapcfg = fullfile(request.brainnet_root, 'BrainNet_MapCfg.m');
expected_whitebg = fullfile(request.compat_dir, 'whitebg.m');
assert(strcmpi(which('BrainNet_MapCfg'), expected_mapcfg), ...
    'BrainNet_MapCfg does not resolve inside the configured BrainNet root.');
assert(strcmpi(which('whitebg'), expected_whitebg), ...
    'The packaged whitebg compatibility shim is not first on path.');

if strcmp(request.mode, 'preflight')
    metadata = struct( ...
        'state', 'passed', ...
        'matlab_release', version('-release'), ...
        'brainnet_mapcfg', which('BrainNet_MapCfg'), ...
        'whitebg', which('whitebg'));
    write_json(request.metadata_path, metadata);
    return;
end

assert(strcmp(request.mode, 'render'), 'Unknown BrainNet request mode.');
required_files = {request.surface, request.node, request.edge, request.viewer_config};
for k = 1:numel(required_files)
    assert(exist(required_files{k}, 'file') == 2, ...
        ['Missing BrainNet input: ' required_files{k}]);
end
close all force;
clear global EC FLAG surf;
cleanup_figures = onCleanup(@() close('all', 'force'));
h = BrainNet_MapCfg( ...
    request.surface, request.node, request.edge, request.viewer_config);
assert(ishandle(h), 'BrainNet_MapCfg did not return a figure handle.');
drawnow;
text_handles = findall(h, 'Type', 'text');
set(text_handles, 'Color', [0 0 0]);
axes_handles = findall(h, 'Type', 'axes');
for k = 1:numel(axes_handles)
    set(axes_handles(k), ...
        'XColor', [0 0 0], 'YColor', [0 0 0], 'ZColor', [0 0 0]);
end
drawnow;
global EC;
EC.img.width = request.width_pixels;
EC.img.height = request.height_pixels;
EC.img.dpi = request.dpi;
set(h, 'PaperPositionMode', 'manual');
set(h, 'PaperUnits', 'inch');
set(h, 'PaperPosition', ...
    [1 1 request.width_pixels / request.dpi request.height_pixels / request.dpi]);
print(h, request.output, '-dpng', ['-r' num2str(request.dpi)]);
assert(exist(request.output, 'file') == 2, ...
    'BrainNet print completed without an output image.');
metadata = struct( ...
    'state', 'succeeded', ...
    'output', request.output, ...
    'width_pixels', request.width_pixels, ...
    'height_pixels', request.height_pixels, ...
    'dpi', request.dpi, ...
    'text_objects', numel(text_handles), ...
    'axes_objects', numel(axes_handles), ...
    'brainnet_mapcfg', which('BrainNet_MapCfg'), ...
    'whitebg', which('whitebg'));
write_json(request.metadata_path, metadata);
close(h);

function write_json(path, value)
fid = fopen(path, 'w');
assert(fid >= 0, ['Could not create metadata: ' path]);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(value));
end
