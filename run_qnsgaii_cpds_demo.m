function run_qnsgaii_cpds_demo()
% QNSGA-II-CPDS  (Minimal Working Version)
% - Encoding: giant string with separators + enable vector z
% - Variation: NSGA-II + Q-learning guided neighborhoods (N1..N7)
% - Objectives: f1 distance, f2 tardiness, f3 #active drones (with soft penalties)
% Author: (for Zoe)
clc; rng(42);

%% ==== Instance (replace with your own data if needed) ====
inst = build_demo_instance();     % coordinates, due times, weights, capacity, energy, etc.
U    = inst.U;                    % #drones
n    = inst.n;                    % #tasks

%% ==== Hyper-parameters ====
NPOP   = 60;          % population size
GEN    = 120;         % generations
pc     = 0.8;         % crossover prob
pm     = 0.4;         % mutation prob (pair-level)
elite_rate = 0.25;    % fraction of elites for Q-learning neighborhoods
penalty_weight = 50;  % penalty scales for violations (capacity/energy)
gamma  = 0.9;         % Q-learning discount
alpha0 = 0.6;         % initial learning rate (linearly decays)
Kact   = 7;           % #actions (N1..N7)
STATE  = 8;           % 2x2x2 states

%% ==== Init population ====
pop = repmat(struct('seq',[],'z',[],'f',[Inf Inf Inf],'rank',Inf,'cd',0,'viol',0), NPOP,1);
for i=1:NPOP
    [seq,z] = init_chromosome(n,U);
    pop(i).seq = seq;
    pop(i).z   = z;
end
pop = evaluate_population(pop, inst, penalty_weight);

%% ==== Q-table and state ====
Q = zeros(STATE, Kact);
cd_prev = mean([pop.cd]);
feas_prev = mean(arrayfun(@(x) x.viol==0, pop))>=0.8;
front_all1 = all([pop.rank]==1);
s = map_state(front_all1, cd_prev, feas_prev);  % initial state index

%% ==== Main loop ====
for gen = 1:GEN
    % epsilon schedule (logistic, max 0.5, inflection at 0.6*GEN)
    eps = 0.5 / (1 + exp(10*((gen) - 0.6*GEN)/GEN));
    alpha = alpha0 * (1 - gen/GEN); alpha = max(alpha, 0.05);

    % ---------- Q-learning neighborhoods on elites ----------
    [pop, Q] = q_neighborhoods(pop, inst, penalty_weight, Q, s, eps, alpha, gamma, elite_rate);

    % ---------- NSGA-II offspring ----------
    offspring = repmat(struct('seq',[],'z',[],'f',[Inf Inf Inf],'rank',Inf,'cd',0,'viol',0), NPOP,1);
    for k=1:NPOP
        p1 = tournament(pop);
        p2 = tournament(pop);
        child = p1;
        if rand < pc
            child.seq = crossover_pairs(p1.seq, p2.seq, U, n); % pair-preserving CO
        end
        if rand < pm
            child.seq = mutate_pairs(child.seq, U, n);
        end
        offspring(k) = child;
    end
    offspring = evaluate_population(offspring, inst, penalty_weight);

    % ---------- Environmental selection ----------
    union = [pop; offspring];
    union = assign_rank_cd(union);
    [~,ord] = sortrows([[union.rank]', -[union.cd]']);
    union = union(ord);
    pop = union(1:NPOP);

    % ---------- Update state ----------
    cd_now = mean([pop.cd]);
    feas_now = mean(arrayfun(@(x) x.viol==0, pop))>=0.8;
    front_all1 = all([pop.rank]==1);
    s = map_state(front_all1, (cd_now>=cd_prev), feas_now);
    cd_prev = cd_now;

    % ---- Logging ----
    if mod(gen,10)==0 || gen==1
        f1 = [pop([pop.rank]==1).f]; f1 = reshape(f1,3,[])';
        hv_est = mean(f1(:,1));     % cheap "proxy" log; replace by HV if needed
        fprintf('Gen %3d | rank1=%d | avgCD=%.3f | feas>=0.8=%d | proxy=%.2f\n',...
            gen, nnz([pop.rank]==1), cd_now, feas_now, hv_est);
    end
end

%% ==== Output PF approx ====
pf = pop([pop.rank]==1);
F = reshape([pf.f],3,[])';
disp('Non-dominated front (f1,f2,f3) (first 10 rows):');
disp(F(1:min(10,size(F,1)),:));

% Plot simple PF projection
figure; scatter(F(:,1),F(:,2),40,F(:,3),'filled');
xlabel('f1 distance'); ylabel('f2 tardiness'); title('QNSGA-II-CPDS (projection)');
colorbar; drawnow;

end % ====== main ======


%% =========================================================
%% ============== Instance & Evaluation ====================
%% =========================================================
function inst = build_demo_instance()
% Larger demo instance (30 drones, 100 pickup-delivery tasks)
inst.U = 30;           % drones available
inst.n = 100;          % pickup-delivery tasks

% Depot + pickups + deliveries coordinates (2D)
% Use several spatial clusters with deterministic jitter for reproducibility
dep = [0,0];
s = rng; rng(1337,'twister');
cluster_centers = [
    -35,  30;
     32,  28;
    -28, -24;
     34, -26;
      0,   0;
    -18,  12;
     16, -14;
    -12, -32;
     24,  18;
    -30,   5];
num_clusters = size(cluster_centers,1);

cluster_idx = randi(num_clusters, inst.n, 1);
offset_pick = randn(inst.n,2) .* [3 3] + rand(inst.n,2).*4 - 2;
P = cluster_centers(cluster_idx,:) + offset_pick;

% Deliveries are offset from pickups towards the depot with additional jitter
to_depot = -0.35 * P + randn(inst.n,2) .* [2.5 2.5];
D = P + to_depot;
rng(s);
inst.dep = dep; inst.P = P; inst.D = D;

% Due times tuned to be tight relative to travel distance from the depot
dist_proxy = vecnorm(P,1,2) + vecnorm(D,1,2);
inst.due = (150 + 0.75 * dist_proxy + 0.5 * vecnorm(P-D,1,2))';

% Diverse payloads to push capacity limits (deterministic pattern)
inst.weight = (3 + mod((1:inst.n)', 5))';

inst.capacity = 24;               % capacity per drone
inst.energyMax = 420;             % energy budget per sortie
inst.speed = 1.0;                 % unit distance per unit time
inst.turn_cost = 0;               % ignore
inst.enable_cost = 10;            % f3 cost per active drone

% Precompute distance matrix between depot & all nodes
% Node index mapping: 0=depot; 1..n => P_i;  n+1..2n => D_i
Nnodes = 1 + 2*inst.n;
C = zeros(Nnodes);
coords = zeros(Nnodes,2);
coords(1,:) = dep;
coords(2:1+inst.n,:) = P;
coords(2+inst.n:end,:) = D;
for i=1:Nnodes
    for j=1:Nnodes
        C(i,j) = sum(abs(coords(i,:)-coords(j,:))); % Manhattan for demo
    end
end
inst.C = C; inst.coords = coords;
end

function pop = evaluate_population(pop, inst, penalty_weight)
for i=1:numel(pop)
    [f, viol] = eval_solution(pop(i).seq, pop(i).z, inst, penalty_weight);
    pop(i).f = f; pop(i).viol = viol;
end
pop = assign_rank_cd(pop);
end

function [f, viol] = eval_solution(seq, z, inst, penalty_weight)
% Decode seq -> routes (cell per drone), then compute f1,f2,f3 + soft penalties
U = numel(z);
routes = split_routes(seq, U);
active = find(z>0);
f1 = 0; f2 = 0; viol = 0;

for u = active
    rt = routes{u};             % list of tokens +i/-i
    [dist, tard, vcount] = simulate_route(rt, inst);
    f1 = f1 + dist;
    f2 = f2 + tard;
    viol = viol + vcount;
end
f3 = numel(active) * inst.enable_cost;

% penalties folded into f1,f2
f1 = f1 + penalty_weight * viol;     % distance-like penalty
f2 = f2 + penalty_weight * 0;        % (you can add lateness penalties here if needed)
f = [f1, f2, f3];
end

function [dist, tard, vcount] = simulate_route(rt, inst)
% Simulate one drone route: depot(1) -> rt -> depot(1)
n = inst.n;
cap = inst.capacity;
Emax = inst.energyMax;
speed = inst.speed;

% Build node sequence including depot IDs from index mapping
% map: +i -> node id = 1+i ;  -i -> id = 1+n+i
seq = [1, arrayfun(@(x) map_token(x,n), rt), 1];
dist = 0; time = 0; energy = 0;
load = 0; tard = 0; vcount = 0;
carrying = false(1,n);

for k=1:numel(seq)-1
    a = seq(k); b = seq(k+1);
    step = inst.C(a,b);
    dist = dist + step;
    time = time + step/speed;
    energy = energy + step*(1+0.1*load); % crude energy model

    % visit effect
    tok = inv_map(seq(k+1), n);
    if tok>0 % pickup
        load = load + inst.weight(tok);
        carrying(tok) = true;
        if load>cap, vcount = vcount+1; end
    elseif tok<0 % delivery
        i = -tok;
        % tardiness only at deliveries
        due = inst.due(i);
        if time>due, tard = tard + (time-due); end
        load = load - inst.weight(i);
        carrying(i) = false;
    end

    if energy>Emax
        vcount = vcount + 1;
        energy = 0; % soft reset
    end
end
end

function id = map_token(tok, n)
if tok>0, id = 1 + tok;
else      id = 1 + n + abs(tok);
end
end
function tok = inv_map(id, n)
% inverse of map_token for the *next* node id
if id==1, tok = 0; return; end
if id<=1+n, tok = id-1; else, tok = -(id-1-n); end
end


%% =========================================================
%% ============== NSGA-II Essentials =======================
%% =========================================================
function pop = assign_rank_cd(pop)
% fast non-dominated sort
F = fast_nondominated_sort(pop);
for r=1:numel(F)
    idx = F{r};
    for t=1:numel(idx), pop(idx(t)).rank = r; end
    pop = crowding_distance(pop, idx);
end
end

function F = fast_nondominated_sort(pop)
m = numel(pop);
S = cell(m,1); n_dom = zeros(m,1);
F = {};
F{1} = [];
for p=1:m
    S{p} = [];
    n_dom(p) = 0;
    for q=1:m
        if dominates(pop(p).f, pop(q).f)
            S{p} = [S{p} q];
        elseif dominates(pop(q).f, pop(p).f)
            n_dom(p) = n_dom(p) + 1;
        end
    end
    if n_dom(p)==0
        pop(p).rank=1; F{1}=[F{1} p];
    end
end
i=1;
while ~isempty(F{i})
    H = [];
    for p=F{i}
        for q=S{p}
            n_dom(q)=n_dom(q)-1;
            if n_dom(q)==0
                pop(q).rank=i+1;
                H=[H q];
            end
        end
    end
    i=i+1; F{i}=H;
end
end

function b = dominates(fa, fb)
b = all(fa<=fb) && any(fa<fb);
end

function pop = crowding_distance(pop, idx)
if isempty(idx), return; end
F = pop(idx);
m = numel(idx); M = 3;
cd = zeros(m,1);
for obj=1:M
    vals = arrayfun(@(x) x.f(obj), F);
    [sv,ord] = sort(vals);
    cd(ord(1)) = inf; cd(ord(end))=inf;
    minv = sv(1); maxv = sv(end);
    if maxv>minv
        for i=2:m-1
            cd(ord(i)) = cd(ord(i)) + (sv(i+1)-sv(i-1))/(maxv-minv);
        end
    end
end
for i=1:m, pop(idx(i)).cd = cd(i); end
end

function a = tournament(pop)
% binary tournament (rank, then cd)
i = randi(numel(pop)); j = randi(numel(pop));
a = better(pop(i), pop(j));
end

function x = better(a,b)
if a.rank<b.rank, x=a; elseif a.rank>b.rank, x=b;
else % larger cd is better
    if a.cd>=b.cd, x=a; else, x=b; end
end
end


%% =========================================================
%% ============== Encoding & Variation =====================
%% =========================================================
function [seq, z] = init_chromosome(n,U)
% Randomly assign each pair to a drone; within drone random order (Pi before Di)
seg = cell(1,U);
for i=1:n
    u = randi(U);
    % random place in segment u; ensure Pi before Di
    seg{u} = [seg{u}, +i, -i];
end
% shuffle inside each segment by pair-blocks
for u=1:U
    seg{u} = shuffle_by_pair(seg{u});
end
seq = join_segments(seg);  % with 0 separators
z = ones(1,U);             % all active initially
end

function out = shuffle_by_pair(v)
% keep Pi before Di, but shuffle pair-blocks
pairs = extract_blocks(v);
if isempty(pairs)
    out = v;
    return;
end
ord = randperm(numel(pairs));
out = [pairs{ord}];
end

function segs = split_routes(seq, U)
% split by 0 into U segments (pad empties)
chunks = split_by_zero(seq);
segs = cell(1,U);
for u=1:U
    if u<=numel(chunks), segs{u}=chunks{u};
    else, segs{u}=[];
    end
end
end

function seq = join_segments(segs)
seq = [];
for u=1:numel(segs)
    if u>1, seq = [seq, 0]; end %#ok<AGROW>
    seq = [seq, segs{u}];   %#ok<AGROW>
end
end

function parts = split_by_zero(seq)
if isempty(seq)
    parts = {};
    return;
end
zpos = [find(seq==0), numel(seq)+1];
start=1; parts={};
for i=1:numel(zpos)
    stop = zpos(i)-1;
    if stop>=start
        parts{end+1} = seq(start:stop); %#ok<AGROW>
    else
        parts{end+1} = [];
    end
    start = zpos(i)+1;
end
end

function blocks = extract_blocks(seg)
% Return cell array of [Pi, ..., Di] per pair while preserving order
if isempty(seg)
    blocks = {};
    return;
end
max_id = max([0, abs(seg(seg~=0))]);
start_idx = zeros(1, max_id);
blocks = {};
for idx = 1:numel(seg)
    tok = seg(idx);
    if tok>0
        if tok>numel(start_idx)
            start_idx(tok) = 0;
        end
        start_idx(tok) = idx;
    elseif tok<0
        id = -tok;
        if id<=numel(start_idx)
            s = start_idx(id);
            if s>0
                blocks{end+1} = seg(s:idx); %#ok<AGROW>
                start_idx(id) = 0;
            end
        end
    end
end
end

function child = crossover_pairs(s1,s2,U,n)
% Pair-preserving crossover:
% For each pair i, pick parent (50/50) to inherit its drone and relative order.
seg1 = split_by_zero(s1); seg2 = split_by_zero(s2);
owner = zeros(1,n);  % which drone
for i=1:n
    if rand<0.5, src = seg1; else, src = seg2; end
    [u,~] = find_pair(src, i);
    if isempty(u), [u,~] = find_pair(seg1, i); end
    if isempty(u), [u,~] = find_pair(seg2, i); end
    if isempty(u), u = randi(U); end
    owner(i)=u;
end
% Reconstruct per drone by following parent-1 order then fill missing from parent-2
segs = cell(1,U); [segs{:}] = deal([]);
fill_order = [seg1, seg2];
for p=1:numel(fill_order)
    seg = fill_order{p};
    if isempty(seg), continue; end
    pairs = extract_blocks(seg);
    for k=1:numel(pairs)
        i = abs(pairs{k}(1));   % pair id
        u = owner(i);
        if ~ismember(+i, segs{u}) && ~ismember(-i, segs{u})
            segs{u} = [segs{u}, +i, -i]; %#ok<AGROW>
        end
    end
end
child = join_segments(segs);
end

function [u, pos] = find_pair(segs, i)
for u=1:numel(segs)
    pos = find(segs{u}==+i,1); pos2 = find(segs{u}== -i,1);
    if ~isempty(pos) && ~isempty(pos2), return; end
end
u=[]; pos=[];
end

function seq = mutate_pairs(seq,U,n)
% Simple mutation: swap two pair-blocks in a random segment
segs = split_by_zero(seq);
u = randi(U);
pairs = extract_blocks(segs{u});
if numel(pairs)>=2
    id = randperm(numel(pairs),2);
    pairs([id(1), id(2)]) = pairs([id(2), id(1)]);
    segs{u} = [pairs{:}];
end
seq = join_segments(segs);
end


%% =========================================================
%% ============== Q-learning & Neighborhoods ===============
%% =========================================================
function s = map_state(front_all1, cd_up, feas_high)
% 2x2x2 -> index 1..8
a = front_all1>0; b = cd_up>0; c = feas_high>0;
s = 1 + a*4 + b*2 + c*1;  % simple mapping
end

function [pop, Q] = q_neighborhoods(pop, inst, penalty_weight, Q, s, eps, alpha, gamma, elite_rate)
% Apply neighborhoods on a subset; update Q-table in-place (passed by value)
N = numel(pop);
k_elite = max(1, round(elite_rate*N));
r1 = [pop.rank]; [~,ord]=sort(r1);
candidates = ord(1:k_elite);
% epsilon-greedy action
if rand < eps, a = randi(size(Q,2));
else           [~,a]=max(Q(s,:));
end

% normalization helpers for reward
allF = reshape([pop.f],3,[])';
fmin = min(allF,[],1); fmax = max(allF,[],1);

for idx = candidates(:)'
    parent = pop(idx);
    child = parent;
    % apply action a
    child.seq = apply_action(a, child.seq, child.z, inst);
    % eval
    [child.f, child.viol] = eval_solution(child.seq, child.z, inst, penalty_weight);
    % accept if child better or nondominated wrt parent
    if dominates(child.f, parent.f) || ~dominates(parent.f, child.f)
        pop(idx) = child;
    end
    % reward against parent→child
    r = reward_multi(parent.f, child.f, fmin, fmax);
    % one-step TD update
    s_next = s; % placeholder (can be replaced by actual next state)
    Q(s,a) = (1-alpha)*Q(s,a) + alpha*( r + gamma*max(Q(s_next,:)) );
end
end

function r = reward_multi(f_old, f_new, fmin, fmax)
% Three-way: dominate / dominated / non-dominated
if dominates(f_new, f_old)
    base = 2;
    gain = sum( (fmax - min(f_new,fmax)) ./ max(fmax - fmin, 1e-9) );
    r = base + gain;
elseif dominates(f_old, f_new)
    base = -0.5;
    loss = sum( (min(f_old,fmax) - f_new) ./ max(fmax - fmin, 1e-9) );
    r = base - 0.5*max(loss,0);
else
    base = 1;
    pos = sum( max(0, (fmax - f_new) ./ max(fmax - fmin, 1e-9) - ...
                       (fmax - f_old) ./ max(fmax - fmin, 1e-9)) );
    neg = sum( max(0, (f_new - f_old) ./ max(fmax - fmin, 1e-9)) );
    r = base + pos - 1.2*neg;
end
% clip
r = max(min(r,3), -3);
end


%% ------------------- Neighborhoods N1..N7 -------------------
% The seven neighborhood actions below are the "moves" that Q-learning
% selects from.  Each action perturbs the encoded chromosome in a
% different manner so that the search can explore complementary routing
% patterns.  A quick summary:
%   N1 Pair Relocate        – remove one pickup-delivery block from a route
%                             and reinsert it at another position of the
%                             same route to locally tweak the sequence.
%   N2 Pair Swap            – exchange the positions of two blocks within a
%                             route to test alternative precedence orders.
%   N3 Cross Exchange       – trade one block between two different drones
%                             so that work can migrate across vehicles.
%   N4 Or-Opt (k blocks)    – lift a string of k consecutive blocks and
%                             relocate the entire string within the same
%                             route, mimicking an Or-opt move.
%   N5 Two-Opt on Blocks    – reverse the order of all blocks between two
%                             indices, analogous to the classical 2-opt
%                             move but applied to whole pickup-delivery
%                             blocks.
%   N6 Toggle Activate      – heuristically move tasks between the most
%                             and least loaded routes to either activate an
%                             idle drone or deactivate an over-fragmented
%                             one.
%   N7 Energy-Aware Reorder – sort the blocks of one route by a composite
%                             key (heavy payload first, tight due time
%                             first, near-depot first) to encourage energy
%                             friendly dispatching.
% Detailed in-function comments explain the mechanics and motivation of
% each move.
function seq2 = apply_action(a, seq, z, inst)
switch a
    case 1, seq2 = N1_pair_relocate(seq);
    case 2, seq2 = N2_pair_swap(seq);
    case 3, seq2 = N3_cross_exchange(seq);
    case 4, seq2 = N4_or_opt(seq, 2);      % k=2
    case 5, seq2 = N5_two_opt_blocks(seq);
    case 6, seq2 = N6_toggle_activate(seq, z); % simple deactivate/activate
    case 7, seq2 = N7_energy_aware_reorder(seq, inst);
    otherwise, seq2 = seq;
end
end

function seq = N1_pair_relocate(seq)
% === Action N1: Pair Relocate ===
% Pick one drone route at random, remove a single pickup-delivery block
% from that route, then insert the block back into another randomly chosen
% position of the same route.  This is a light-weight local search move
% that can shorten detours or relieve resource spikes without changing the
% responsible drone.
segs = split_by_zero(seq);
u = randi(numel(segs));                    % pick a route/drone
pairs = extract_blocks(segs{u});
if isempty(pairs), return; end             % nothing to relocate
k = randi(numel(pairs));
blk = pairs{k};                            % block chosen for relocation
pairs(k) = [];                             % temporarily remove block
pos = randi(numel(pairs)+1);               % choose new insertion slot
pairs = insert_cells(pairs, {blk}, pos);   % splice block back in
segs{u} = [pairs{:}];
seq = join_segments(segs);
end

function seq = N2_pair_swap(seq)
% === Action N2: Pair Swap ===
% Select a single route and swap two randomly chosen pickup-delivery
% blocks.  The drone assignment remains untouched but the pair precedence
% changes, letting the algorithm try alternative visit orders that might
% reduce travel or tardiness.
segs = split_by_zero(seq);
u = randi(numel(segs));
pairs = extract_blocks(segs{u}); m = numel(pairs);
if m<2, return; end                        % at least two blocks required
id = randperm(m,2);
pairs([id(1), id(2)]) = pairs([id(2), id(1)]);
segs{u} = [pairs{:}];
seq = join_segments(segs);
end

function seq = N3_cross_exchange(seq)
% === Action N3: Cross Exchange ===
% Choose two distinct drones and exchange one block between them.  This is
% how work gets redistributed across the fleet: one drone relinquishes a
% task while another takes over, potentially balancing load and energy
% budgets.  After the swap both routes reinsert the incoming block at random
% positions to generate new sequencing opportunities.
segs = split_by_zero(seq);
U = numel(segs);
if U<2, return; end
uv = randperm(U,2); u=uv(1); v=uv(2);      % pick two distinct routes
pu = extract_blocks(segs{u});
pv = extract_blocks(segs{v});
if isempty(pu) || isempty(pv), return; end % both need at least one block
ku = randi(numel(pu)); kv = randi(numel(pv));
bu = pu{ku}; bv = pv{kv};                  % exchange candidates
pu(ku) = []; pv(kv) = [];                  % remove the chosen blocks
posu = randi(numel(pu)+1);                 % reinsertion slots
pvu = randi(numel(pv)+1);
pu = insert_cells(pu, {bv}, posu);         % drone u receives block bv
pv = insert_cells(pv, {bu}, pvu);          % drone v receives block bu
segs{u} = [pu{:}];
segs{v} = [pv{:}];
seq = join_segments(segs);
end

function seq = N4_or_opt(seq, k)
% === Action N4: Or-Opt with k blocks ===
% Generalises the classical Or-opt neighborhood: we pick k consecutive
% blocks (k defaults to 2) from a route, remove the entire subsequence, and
% reinsert it elsewhere in the same route.  Moving a small "string" of
% tasks together is effective for re-routing trucks/vehicles that must keep
% pickup-delivery precedence intact.
segs = split_by_zero(seq);
u = randi(numel(segs));
pairs = extract_blocks(segs{u}); m = numel(pairs);
if m<k || k==0, return; end
i = randi(m-k+1);                           % starting index of subsequence
blk_cells = pairs(i:i+k-1);                 % capture the consecutive block
pairs(i:i+k-1) = [];                        % remove from current position
pos = randi(numel(pairs)+1);                % insertion point elsewhere
pairs = insert_cells(pairs, blk_cells, pos);
segs{u} = [pairs{:}];
seq = join_segments(segs);
end

function seq = N5_two_opt_blocks(seq)
% === Action N5: Two-Opt on Blocks ===
% Inspired by 2-opt for TSP, this move chooses two cut positions i < j in
% one route and reverses the order of the blocks between them.  Because
% blocks already enforce pickup-before-delivery, reversing the block list
% simply explores the opposite traversal direction for that path segment.
segs = split_by_zero(seq);
u = randi(numel(segs));
pairs = extract_blocks(segs{u}); m=numel(pairs);
if m<2, return; end
i = randi(m-1); j = randi([i+1,m]);
pairs(i:j) = pairs(j:-1:i);                 % reverse block sequence
segs{u} = [pairs{:}];
seq = join_segments(segs);
end

function seq = N6_toggle_activate(seq, z)
% === Action N6: Toggle Activate ===
% Encourage flexible fleet usage.  If an idle drone exists (empty route),
% we "activate" it by moving a random block from the busiest drone to the
% idle one.  Otherwise we attempt to "deactivate" one drone by merging its
% tasks into the busiest route, which may reduce the number of active
% drones (and thus objective f3).  The move leaves the binary activation
% vector z untouched because the evaluation function already infers active
% drones from empty/non-empty routes.
segs = split_by_zero(seq); U=numel(segs);
lens = cellfun(@numel, segs);
if any(lens==0)
    % --- Activation branch: populate one empty route ---
    [~,u] = max(lens);                      % currently most loaded route
    v = find(lens==0,1);                    % first empty route to fill
    prs = extract_blocks(segs{u});
    if isempty(prs)
        return;                             % degenerate case: no movable block
    end
    k = randi(numel(prs));
    blk = prs{k};
    prs(k) = [];                            % remove block from busy route
    segs{u} = [prs{:}];
    segs{v} = blk;                          % activate idle drone with block
else
    % --- Deactivation branch: merge light route into heavy route ---
    [~,u] = max(lens);                      % recipient (most capacity)
    [~,v] = min(lens);                      % donor (least workload)
    prs_long = extract_blocks(segs{u});
    prs_short = extract_blocks(segs{v});
    if isempty(prs_short)
        return;                             % nothing to merge
    end
    for k=1:numel(prs_short)
        pos = randi(numel(prs_long)+1);
        prs_long = insert_cells(prs_long, prs_short(k), pos); %#ok<AGROW>
    end
    segs{u} = [prs_long{:}];
    segs{v} = [];
end
seq = join_segments(segs);
end

function seq = N7_energy_aware_reorder(seq, inst)
% === Action N7: Energy-Aware Reorder ===
% Select one drone and sort its blocks by a composite priority key that is
% correlated with energy consumption: heavier payloads first (to drop
% weight sooner), earlier due times first (to avoid tardiness), and shorter
% pickup distance from the depot first (to limit long deadhead legs).  The
% deterministic sort provides a directed improvement move rather than a
% random perturbation.
segs = split_by_zero(seq); U=numel(segs);
if U==0, return; end
u = randi(U);
pairs = extract_blocks(segs{u}); m=numel(pairs);
if m<=1, return; end
key = zeros(m,3);
for k=1:m
    i = abs(pairs{k}(1));
    w = inst.weight(i);
    due = inst.due(i);
    idP = 1+i;                              % pickup node index in matrix
    key(k,:) = [ -w, due, inst.C(1,idP) ];  % sort descending by weight
end
[~,ord] = sortrows(key,[1 2 3]);           % lexicographic sort by key
pairs = pairs(ord);
segs{u} = [pairs{:}];
seq = join_segments(segs);
end

function list = insert_cells(list, items, pos)
% Insert ITEMS (cell array) into LIST at position POS (1..n+1)
if isempty(list)
    list = items;
    return;
end
if pos <= 1
    list = [items, list];
elseif pos > numel(list)
    list = [list, items];
else
    list = [list(1:pos-1), items, list(pos:end)];
end
end
