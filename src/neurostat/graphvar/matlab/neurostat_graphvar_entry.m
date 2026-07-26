% NeuroStat noninteractive GraphVar 2.03a request boundary.
requestPath = getenv('NEUROSTAT_GRAPHVAR_REQUEST');
assert(~isempty(requestPath) && isfile(requestPath), ...
    'NEUROSTAT_GRAPHVAR_REQUEST is missing or invalid.');
Request = jsondecode(fileread(requestPath));
graphvarRoot = char(Request.graphvar_root);
assert(isfolder(graphvarRoot), 'Configured GraphVar root is not a directory.');
addpath(fullfile(graphvarRoot, 'src'));
addpath(fullfile(graphvarRoot, 'src', 'calc'));
addpath(fullfile(graphvarRoot, 'src', 'calc', 'tfun'));
glmPath = which('graphvar_glm');
expectedGlm = fullfile(graphvarRoot, 'src', 'calc', 'graphvar_glm.m');
assert(strcmpi(canonicalPath(glmPath), canonicalPath(expectedGlm)), ...
    'graphvar_glm.m resolved outside the configured GraphVar root.');
startPath = fullfile(graphvarRoot, 'start_GraphVar.m');
startSource = fileread(startPath);
assert(contains(startSource, 'Release= 2.03a'), ...
    'Configured GraphVar installation is not 2.03a.');

if strcmp(Request.mode, 'preflight')
    requiredFunctions = {'jsondecode','readcell','rng','regress','fcdf'};
    for requiredIndex = 1:numel(requiredFunctions)
        assert(~isempty(which(requiredFunctions{requiredIndex})), ...
            'Required MATLAB function is unavailable: %s', ...
            requiredFunctions{requiredIndex});
    end
    hasStatistics = license('test', 'Statistics_Toolbox');
    assert(hasStatistics, 'MATLAB Statistics and Machine Learning Toolbox is required.');
    Metadata = struct( ...
        'matlab_release', version('-release'), ...
        'graphvar_release', '2.03a', ...
        'graphvar_glm', canonicalPath(glmPath), ...
        'graphvar_glm_sha256', fileSha256(glmPath), ...
        'start_graphvar_sha256', fileSha256(startPath), ...
        'statistics_toolbox', logical(hasStatistics), ...
        'required_functions', {requiredFunctions});
    writeJson(char(Request.metadata_path), Metadata);
    clear Request Metadata;
    return;
end
assert(strcmp(Request.mode, 'run'), 'Unknown NeuroStat GraphVar request mode.');
workspace = char(Request.workspace);
Settings = load(fullfile(workspace, 'Workspace.mat'));
variables = readcell(fullfile(workspace, Settings.variableSheet), ...
    'Delimiter', ';', 'TextType', 'char');
assert(isequal(variables(1,:), {'ObservationID','arm','subject','session'}), ...
    'Variables.csv has an unexpected header.');
observations = variables(2:end,:);
nObservations = size(observations, 1);
relativeFiles = cellstr(Settings.files_rel);
assert(numel(relativeFiles) == nObservations, ...
    'Workspace file/row cardinality differs.');
selected = double(Request.atlas_indices_1based(:)');
nNodes = numel(selected);
lowerMask = ~triu(true(nNodes));
nLinks = nnz(lowerMask);
outcomes = zeros(nObservations, nLinks);
for observation = 1:nObservations
    matrixPath = fullfile(workspace, 'data', 'CorrMatrix', ...
        relativeFiles{observation});
    Content = load(matrixPath, Settings.fieldName);
    assert(isfield(Content, Settings.fieldName), ...
        'Configured matrix field is absent.');
    matrix = Content.(Settings.fieldName);
    assert(ismatrix(matrix) && size(matrix,1) == size(matrix,2) && ...
        max(selected) <= size(matrix,1), 'Connectivity matrix dimensions differ.');
    matrix = matrix(selected, selected);
    outcomes(observation,:) = matrix(lowerMask);
end

[~, RFUN, ~, LAB, LLAB, LLEV, LASSIGN, LTYPE, ...
    MMEANS, J, N, DDF] = ...
    graphvar_glm(observations, [], 3, [], [], 2, 1, {'arm'});
armAssignment = find(strcmp(LAB, 'arm'));
assert(isscalar(armAssignment), 'The arm factor was not uniquely identified.');
effects = LASSIGN == armAssignment;
assert(any(effects), 'No arm hypotheses were formed.');
effectLabels = string(LLAB(effects));
assert(sum(effectLabels == string(Request.expected_effect_label)) == 1, ...
    'Configured GraphVar effect label is not unique.');
rng(double(Request.seed), 'twister');
[P, NP, F, NF, B, NB] = RFUN(outcomes, double(Request.permutations));

Result = struct();
Result.N = N;
Result.J = J(effects);
Result.DDF = DDF;
Result.P = shapeHalfObserved(P(effects,:), lowerMask);
Result.NP = shapeHalfObserved(NP(effects,:), lowerMask);
Result.F = shapeHalfObserved(F(effects,:), lowerMask);
Result.NF = shapeHalfNull(NF(effects,:,:), lowerMask);
Result.B = shapeHalfObserved(B(effects,:), lowerMask);
Result.NB = shapeHalfNull(NB(effects,:,:), lowerMask);
Result.SE = zeros(size(Result.P));
Result.LAB = LLAB(effects);
Result.TYPE = LTYPE(effects);
groupMeans = MMEANS(effects);
groupLevels = LLEV(effects);
meanMask = ~cellfun(@isempty, groupMeans);
Result.LEV = 0;
Result.LLEV = 0;
Result.MEAN = 0;
if any(meanMask)
    Result.LEV = groupMeans(meanMask);
    Result.LLEV = groupLevels;
    Result.MEAN = Result.B(:,:,meanMask);
end
resultPath = char(Request.result_path);
save(resultPath, '-struct', 'Result', '-v7.3');
Metadata = struct( ...
    'state', 'succeeded', ...
    'seed', double(Request.seed), ...
    'permutations', double(Request.permutations), ...
    'observations', nObservations, ...
    'result_sha256', fileSha256(resultPath));
writeJson(char(Request.metadata_path), Metadata);

function shaped = shapeHalfObserved(values, lowerMask)
nNodes = size(lowerMask, 1);
nEffects = size(values, 1);
shaped = zeros(nNodes, nNodes, nEffects);
for effect = 1:nEffects
    matrix = zeros(nNodes);
    matrix(lowerMask) = values(effect,:);
    shaped(:,:,effect) = matrix + matrix';
end
% MATLAB v7.3 stores this as physical (effects,nodes,nodes) in HDF5.
end

function shaped = shapeHalfNull(values, lowerMask)
nNodes = size(lowerMask, 1);
nEffects = size(values, 1);
nPermutations = size(values, 3);
shaped = zeros(nNodes, nNodes, nEffects, nPermutations);
for effect = 1:nEffects
    matrices = zeros(nNodes * nNodes, nPermutations);
    matrices(lowerMask(:),:) = reshape( ...
        values(effect,:,:), nnz(lowerMask), nPermutations);
    matrices = reshape(matrices, nNodes, nNodes, nPermutations);
    matrices = matrices + permute(matrices, [2 1 3]);
    shaped(:,:,effect,:) = reshape(matrices, nNodes, nNodes, 1, nPermutations);
end
% MATLAB v7.3 stores this as physical (permutations,effects,nodes,nodes).
end

function result = canonicalPath(path)
result = char(java.io.File(path).getCanonicalPath());
end

function digest = fileSha256(path)
engine = java.security.MessageDigest.getInstance('SHA-256');
fileObject = java.io.File(path);
content = java.nio.file.Files.readAllBytes(fileObject.toPath());
bytes = typecast(engine.digest(content), 'uint8');
digest = lower(reshape(dec2hex(bytes, 2).', 1, []));
end

function writeJson(path, value)
temporary = [path '.tmp'];
handle = fopen(temporary, 'w', 'n', 'UTF-8');
assert(handle >= 0, 'Could not open metadata file.');
cleanup = onCleanup(@() fclose(handle));
fprintf(handle, '%s\n', jsonencode(value));
clear cleanup;
movefile(temporary, path, 'f');
end
