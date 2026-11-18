/* Interactive model-state graph with collapsible annotation parts (plain JS, SVG). */
var XG = (function(){
  var SVGNS = 'http://www.w3.org/2000/svg';
  var PALETTE = ['#4e79a7','#e8853a','#d9534f','#5aa9a3','#59a14f','#c9a227','#b07aa1','#e17c9a','#9c755f','#7f7fd5','#2a9d8f','#8d99ae'];
  var REDUCED = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  var BUNDLE_MIN = 12;
  var STORE = {};   // moved positions, per explanation, for the session (and localStorage when allowed)
  function memLoad(key){
    if(!key) return {states:{}, nodes:{}, fns:{}};
    if(STORE[key]) return STORE[key];
    var m=null; try{ var raw=localStorage.getItem('ch7-xg-'+key); if(raw) m=JSON.parse(raw); }catch(e){ m=null; }
    if(!m || typeof m!=='object' || !m.states) m={states:{}, nodes:{}, fns:{}};
    STORE[key]=m; return m;
  }
  function memSave(key, m){
    if(!key) return; STORE[key]=m;
    try{ var empty=!Object.keys(m.nodes).length && !Object.keys(m.fns).length;
      if(empty) localStorage.removeItem('ch7-xg-'+key); else localStorage.setItem('ch7-xg-'+key, JSON.stringify(m)); }catch(e){}
  }

  function el(tag, attrs, parent){
    var e = document.createElementNS(SVGNS, tag);
    if(attrs){ for(var k in attrs){ if(attrs[k]!=null) e.setAttribute(k, attrs[k]); } }
    if(parent) parent.appendChild(e);
    return e;
  }
  function fmt(v){ if(v==null) return 'n/a'; var a=Math.abs(v); if(a!==0&&(a<1e-3||a>=1e5)) return v.toExponential(2); return String(Math.round(v*1e4)/1e4); }

  // ---------- model ----------
  function prepare(g){
    var M = {nodes:{}, order:[], edges:[], parts:{}, partList:[], outputs:g.outputs||[], inputs:g.inputs||[], titles:g.titles||{}, readings:g.readings||{}};
    g.nodes.forEach(function(n, i){ M.nodes[n[0]] = {id:n[0], kind:n[1], name:n[2], act:n[3], bias:n[4], idx:i}; M.order.push(n[0]); });
    g.edges.forEach(function(e){ if(M.nodes[e[0]] && M.nodes[e[1]]) M.edges.push({s:e[0], t:e[1], w:e[2], on:!!e[3]}); });
    g.anns.forEach(function(a, i){
      M.parts[a.name] = {name:a.name, aid:a.id, parent:a.parent && a.parent!==a.name ? a.parent : null, children:[],
        own:new Set(a.nodes), entries:new Set(a.entries), exits:new Set(a.exits), color:PALETTE[i % PALETTE.length], idx:i,
        title:(M.titles[a.id]||null), reading:(M.readings[a.id]||null)};
    });
    Object.keys(M.parts).forEach(function(k){ var p=M.parts[k]; if(p.parent && M.parts[p.parent]) M.parts[p.parent].children.push(k); else p.parent=null; });
    function depth(p, seen){ if(!p.parent) return 0; if(seen[p.name]) return 0; seen[p.name]=1; return 1+depth(M.parts[p.parent], seen); }
    var memo = {};
    function all(name, seen){
      if(memo[name]) return memo[name];
      seen = seen||{}; if(seen[name]) return new Set(); seen[name]=1;
      var p=M.parts[name], s=new Set(p.own);
      p.children.forEach(function(c){ all(c, seen).forEach(function(x){ s.add(x); }); });
      memo[name]=s; return s;
    }
    Object.keys(M.parts).forEach(function(k){
      var p=M.parts[k]; p.depth=depth(p, {}); p.all=all(k);
      p.hidden=new Set(); p.all.forEach(function(x){ if(!p.entries.has(x) && M.nodes[x]) p.hidden.add(x); });
    });
    M.partList = Object.keys(M.parts).map(function(k){ return M.parts[k]; }).sort(function(a,b){ return a.depth-b.depth || a.idx-b.idx; });
    M.maxDepth = M.partList.reduce(function(m,p){ return Math.max(m,p.depth); }, 0);
    M.home = {};
    M.order.forEach(function(id){
      var best=null; M.partList.forEach(function(p){ if(p.own.has(id) && !p.entries.has(id)) best=p; });
      if(!best) M.partList.forEach(function(p){ if(p.hidden.has(id)) best=p; });
      M.home[id]=best?best.name:null;
    });
    M.label = function(p){ return p.title || p.name; };
    return M;
  }

  function levelSet(M, L){ var s=new Set(); M.partList.forEach(function(p){ if(p.depth < L) s.add(p.name); }); return s; }

  // ---------- view: collapse + aggregate ----------
  function view(M, expanded, showOff){
    var rep = {};
    M.order.forEach(function(id){
      rep[id]=id;
      for(var i=0;i<M.partList.length;i++){ var p=M.partList[i]; if(!expanded.has(p.name) && p.hidden.has(id)){ rep[id]='F:'+p.name; break; } }
    });
    var vn = {}, vorder=[];
    M.order.forEach(function(id){
      var r=rep[id]; if(vn[r]) return;
      if(r.slice(0,2)==='F:'){ var p=M.parts[r.slice(2)]; vn[r]={id:r, kind:'function', part:p.name}; }
      else { var n=M.nodes[id]; vn[r]={id:r, kind:n.kind, node:n}; }
      vorder.push(r);
    });
    var agg = {};
    M.edges.forEach(function(e){
      if(!showOff && (!e.on || e.w===0)) return;
      var a=rep[e.s], b=rep[e.t]; if(a===b) return;
      var k=a+'>'+b; var x=agg[k] || (agg[k]={s:a, t:b, list:[], sum:0, pos:0, neg:0, off:0});
      x.list.push(e); x.sum+=Math.abs(e.w||0); if(!e.on||e.w===0) x.off++; else if(e.w>0) x.pos++; else x.neg++;
    });
    var ve = Object.keys(agg).map(function(k){ return agg[k]; });
    // drop isolated nodes that only connect through hidden edges (e.g. pruned inputs)
    var deg={}; ve.forEach(function(e){ deg[e.s]=1; deg[e.t]=1; });
    vorder = vorder.filter(function(id){ return deg[id] || vn[id].kind==='function' || vn[id].kind==='output'; });
    // bundle many inputs that feed only one collapsed function node
    var outs={}; ve.forEach(function(e){ (outs[e.s]=outs[e.s]||[]).push(e); });
    var byTarget={};
    vorder.forEach(function(id){
      var v=vn[id]; if(v.kind!=='input' && v.kind!=='split_input') return;
      var o=outs[id]||[]; if(!o.length) return;
      var t=o[0].t; if(vn[t].kind!=='function') return;
      for(var i=1;i<o.length;i++){ if(o[i].t!==t) return; }
      (byTarget[t]=byTarget[t]||[]).push(id);
    });
    Object.keys(byTarget).forEach(function(t){
      var ids=byTarget[t]; if(ids.length < BUNDLE_MIN) return;
      var bid='B:'+t; vn[bid]={id:bid, kind:'bundle', members:ids, target:t};
      var set=new Set(ids); var be={s:bid, t:t, list:[], sum:0, pos:0, neg:0, off:0};
      ve = ve.filter(function(e){ if(set.has(e.s)){ be.list=be.list.concat(e.list); be.sum+=e.sum; be.pos+=e.pos; be.neg+=e.neg; be.off+=e.off; return false; } return true; });
      ve.push(be);
      var at=vorder.indexOf(ids[0]); vorder=vorder.filter(function(id){ return !set.has(id); }); vorder.splice(Math.max(0,at),0,bid);
    });
    return {nodes:vn, order:vorder, edges:ve};
  }

  // ---------- layout: layers by longest path, barycentre ordering ----------
  function size(v, M){
    if(v.kind==='function'){ var t=M.label(M.parts[v.part]); var w=Math.min(230, Math.max(96, t.length*6.6+34)); return {w:w, h:30}; }
    if(v.kind==='bundle') return {w:84, h:24};
    return {w:14, h:14};
  }
  function layout(V, M){
    var ids=V.order, pred={}, succ={};
    ids.forEach(function(id){ pred[id]=[]; succ[id]=[]; });
    V.edges.forEach(function(e){ if(pred[e.t] && succ[e.s]){ succ[e.s].push(e.t); pred[e.t].push(e.s); } });
    var isIn=function(id){ var k=V.nodes[id].kind; return k==='input'||k==='split_input'||k==='bundle'; };
    var isOut=function(id){ return V.nodes[id].kind==='output'; };
    // longest path with cycle guard (DFS)
    var layer={}, state={};
    function lp(id){
      if(state[id]===2) return layer[id]; if(state[id]===1) return 0;
      state[id]=1; var L=isIn(id)?0:1;
      if(!isIn(id)) pred[id].forEach(function(p){ L=Math.max(L, lp(p)+1); });
      state[id]=2; layer[id]=L; return L;
    }
    ids.forEach(lp);
    var maxL=0; ids.forEach(function(id){ maxL=Math.max(maxL, layer[id]); });
    ids.forEach(function(id){ if(isOut(id)) layer[id]=Math.max(maxL,1); });
    maxL=0; ids.forEach(function(id){ maxL=Math.max(maxL, layer[id]); });
    var layers=[]; for(var i=0;i<=maxL;i++) layers.push([]);
    ids.forEach(function(id){ layers[layer[id]].push(id); });
    var pos={};
    function setPos(){ layers.forEach(function(Ly){ Ly.forEach(function(id,i){ pos[id]=i; }); }); }
    setPos();
    function bary(id, nb){ var s=0,n=0; nb[id].forEach(function(x){ if(pos[x]!=null){ s+=pos[x]/Math.max(1,layers[layer[x]].length); n++; } }); return n? s/n : null; }
    for(var it=0; it<6; it++){
      var down = it%2===0;
      var seq = down? layers.slice(1) : layers.slice(0,-1).reverse();
      seq.forEach(function(Ly){
        var nb = down? pred : succ;
        var keyed = Ly.map(function(id,i){ var b=bary(id,nb); return {id:id, b:b==null? i/Math.max(1,Ly.length): b}; });
        keyed.sort(function(a,b){ return a.b-b.b; });
        keyed.forEach(function(k,i){ Ly[i]=k.id; });
        setPos();
      });
    }
    // Parts as horizontal slabs: every expanded part gets its own vertical band across all layers,
    // nested inside its parent's band, so the bands never overlap. Inputs and outputs sit outside.
    function chain(id){
      var v=V.nodes[id], p=null;
      if(v.kind==='function'){ var q=M.parts[v.part]; p=q.parent? M.parts[q.parent] : null; }
      else if(v.node && M.home[v.node.id]) p=M.parts[M.home[v.node.id]];
      var c=[]; while(p){ c.unshift(p.name); p=p.parent? M.parts[p.parent] : null; } return c;
    }
    var norm={}; ids.forEach(function(id){ norm[id]=pos[id]/Math.max(1,layers[layer[id]].length-1); });
    var colW=Math.max(170, Math.min(240, 1000/Math.max(1,maxL))), gap=10, PAD=10, LBL=20;
    var root={name:null, kids:{}, items:[], depth:-1};
    ids.forEach(function(id){
      if(isIn(id) || isOut(id)) return;
      var g=root; chain(id).forEach(function(n, d){ g=g.kids[n] || (g.kids[n]={name:n, kids:{}, items:[], depth:d}); });
      g.items.push(id);
    });
    function avg(g){ if(g._avg!=null) return g._avg; var s=0,n=0; g.items.forEach(function(id){ s+=norm[id]; n++; });
      Object.keys(g.kids).forEach(function(k){ var a=avg(g.kids[k]); s+=a; n++; }); g._avg=n? s/n : 0.5; return g._avg; }
    function hOf(id){ return size(V.nodes[id],M).h; }
    var P={};
    function place(g, y0){   // returns height
      var looseByLayer={}; g.items.forEach(function(id){ (looseByLayer[layer[id]]=looseByLayer[layer[id]]||[]).push(id); });
      var looseH=0; Object.keys(looseByLayer).forEach(function(L){ var arr=looseByLayer[L]; arr.sort(function(a,b){ return norm[a]-norm[b]; });
        looseH=Math.max(looseH, arr.reduce(function(s,id){ return s+hOf(id)+gap; },0)); });
      var blocks=Object.keys(g.kids).map(function(k){ return {kind:'g', g:g.kids[k], a:avg(g.kids[k])}; });
      if(g.items.length) blocks.push({kind:'loose', a:g.items.reduce(function(s,id){ return s+norm[id]; },0)/g.items.length});
      blocks.sort(function(a,b){ return a.a-b.a; });
      var top = g.name? LBL : 0, y=y0+top;
      blocks.forEach(function(bk, i){
        if(i) y+=PAD;
        if(bk.kind==='loose'){
          Object.keys(looseByLayer).forEach(function(L){ var arr=looseByLayer[L], h=arr.reduce(function(s,id){ return s+hOf(id)+gap; },0), yy=y+(looseH-h)/2;
            arr.forEach(function(id){ var sz=size(V.nodes[id],M); P[id]={x:+L*colW, y:yy+sz.h/2, w:sz.w, h:sz.h, layer:+L}; yy+=sz.h+gap; }); });
          y+=looseH;
        } else { var h=place(bk.g, y); y+=h; }
      });
      var H=y-y0+(g.name? PAD : 0);
      if(g.name) g.box={y0:y0, y1:y0+H};
      return H;
    }
    var H=Math.max(40, place(root, 0));
    // inputs: ordered by the mean height of what they feed, spread evenly over the height
    var ins=ids.filter(isIn), outsL=ids.filter(isOut);
    function meanY(id, nb){ var s=0,n=0; nb[id].forEach(function(x){ if(P[x]){ s+=P[x].y; n++; } }); return n? s/n : H/2; }
    ins.sort(function(a,b){ return meanY(a,succ)-meanY(b,succ) || norm[a]-norm[b]; });
    var inH=ins.reduce(function(s,id){ return s+hOf(id)+gap; },0);
    if(inH>H){ var off=(inH-H)/2; Object.keys(P).forEach(function(id){ P[id].y+=off; });
      (function walk(g){ Object.keys(g.kids).forEach(function(k){ var b=g.kids[k].box; if(b){ b.y0+=off; b.y1+=off; } walk(g.kids[k]); }); })(root); H=inH; }
    var yy=0, spare=Math.max(0, H-inH)/Math.max(1, ins.length);
    // place inputs near their targets but keep order and spacing
    var want=ins.map(function(id){ return meanY(id,succ); }), cur=-Infinity;
    ins.forEach(function(id, i){ var sz=size(V.nodes[id],M); var y=Math.max(want[i], cur+sz.h/2+gap); P[id]={x:0, y:y, w:sz.w, h:sz.h, layer:0}; cur=y+sz.h/2; });
    var over=(cur - H); if(over>0){ ins.forEach(function(id){ P[id].y-=over; }); }
    var minIn=ins.length? P[ins[0]].y - P[ins[0]].h/2 : 0; if(minIn<0){ ins.forEach(function(id){ P[id].y-=minIn; }); }
    outsL.forEach(function(id){ var sz=size(V.nodes[id],M); P[id]={x:layer[id]*colW, y:meanY(id,pred), w:sz.w, h:sz.h, layer:layer[id]}; });
    var groups={}; (function walk(g){ Object.keys(g.kids).forEach(function(k){ groups[k]=g.kids[k].box; walk(g.kids[k]); }); })(root);
    return {P:P, W:maxL*colW, H:H, colW:colW, groups:groups};
  }

  // ---------- rendering ----------
  function edgePath(a, b){
    var x1=a.x + (a.w/2), y1=a.y, x2=b.x - (b.w/2), y2=b.y;
    if(x2 < x1+10){ var mx=(x1+x2)/2; return 'M'+x1+','+y1+' C'+(x1+60)+','+(y1-40)+' '+(x2-60)+','+(y2-40)+' '+x2+','+y2; }
    var c=(x2-x1)*0.45;
    return 'M'+x1+','+y1+' C'+(x1+c)+','+y1+' '+(x2-c)+','+y2+' '+x2+','+y2;
  }
  function edgeColor(e){ if(e.off===e.list.length) return 'var(--g-off)'; if(e.pos && e.neg) return 'var(--g-mix)'; return e.pos? 'var(--g-pos)':'var(--g-neg)'; }
  function nodeLabel(n){ return n.name || n.id; }

  function thumb(g){
    var M=prepare(g), V=view(M, levelSet(M,1), false), Lo=layout(V,M), P=Lo.P;
    var pad=24, W=Lo.W+120+pad*2, H=Lo.H+pad*2;
    var svg=el('svg',{viewBox:(-60-pad)+' '+(-pad)+' '+W+' '+H, width:'132', height:'70', 'aria-hidden':'true', preserveAspectRatio:'xMidYMid meet', class:'gthumb'});
    var maxW=0; V.edges.forEach(function(e){ maxW=Math.max(maxW, e.sum); });
    V.edges.forEach(function(e){ var a=P[e.s], b=P[e.t]; if(!a||!b) return; el('path',{d:edgePath(a,b), fill:'none', stroke:edgeColor(e), 'stroke-width':(1+5*Math.sqrt(e.sum/(maxW||1))).toFixed(2), 'stroke-opacity':0.75}, svg); });
    V.order.forEach(function(id){
      var v=V.nodes[id], p=P[id]; if(!p) return;
      if(v.kind==='function'){ var part=M.parts[v.part]; el('rect',{x:p.x-p.w/2, y:p.y-p.h/2, width:p.w, height:p.h, rx:10, fill:part.color, 'fill-opacity':0.35, stroke:part.color, 'stroke-width':3}, svg); }
      else el('circle',{cx:p.x, cy:p.y, r:v.kind==='output'?9:6, fill:'var(--g-node)', stroke:'var(--ink-3)', 'stroke-width':1.5}, svg);
    });
    return svg;
  }

  function mount(container, g, opts){
    opts=opts||{};
    var M=prepare(g);
    var expanded=levelSet(M,1), showOff=false, selected=null;
    container.replaceChildren();
    container.classList.add('gwrap');
    // toolbar
    var bar=document.createElement('div'); bar.className='gbar';
    function btn(label, fn, title){ var b=document.createElement('button'); b.type='button'; b.textContent=label; if(title) b.title=title; b.addEventListener('click', fn); bar.appendChild(b); return b; }
    btn('Collapse all', function(){ setExpanded(levelSet(M,1)); }, 'The recomposed view: the top-level parts under the root');
    btn('Expand one level', function(){
      var V=view(M, expanded, showOff), d=Infinity;
      V.order.forEach(function(id){ var v=V.nodes[id]; if(v.kind==='function') d=Math.min(d, M.parts[v.part].depth); });
      if(d===Infinity) return; var s=new Set(expanded); M.partList.forEach(function(p){ if(p.depth<=d) s.add(p.name); }); setExpanded(s);
    });
    btn('Expand all', function(){ setExpanded(new Set(M.partList.map(function(p){ return p.name; }))); }, 'The fully decomposed network');
    var lab=document.createElement('label'); lab.className='glevel';
    lab.appendChild(document.createTextNode('Level '));
    var slider=document.createElement('input'); slider.type='range'; slider.min='0'; slider.max=String(M.maxDepth+1); slider.step='1'; slider.value='1';
    slider.setAttribute('aria-label','Level: 0 = one function, '+(M.maxDepth+1)+' = fully decomposed');
    slider.addEventListener('input', function(){ setExpanded(levelSet(M, +slider.value), true); });
    lab.appendChild(slider); var lvOut=document.createElement('span'); lvOut.className='glv'; lab.appendChild(lvOut); bar.appendChild(lab);
    btn('Fit', function(){ fit(true); }, 'Fit the graph to the panel');
    btn('Reset layout', function(){ mem={states:{}, nodes:{}, fns:{}}; memSave(opts.key, mem); draw(); requestAnimationFrame(function(){ fit(true); }); }, 'Forget the positions of moved nodes');
    var offLab=document.createElement('label'); offLab.className='goff'; var off=document.createElement('input'); off.type='checkbox';
    off.addEventListener('change', function(){ showOff=off.checked; draw(); }); offLab.appendChild(off); offLab.appendChild(document.createTextNode(' disabled / zero edges')); bar.appendChild(offLab);
    container.appendChild(bar);
    var stage=document.createElement('div'); stage.className='gstage'; container.appendChild(stage);
    var svg=el('svg',{class:'gsvg', role:'group', 'aria-label':'Model graph: function nodes are collapsed parts; Tab through nodes, Enter expands or collapses'}, stage);
    var vp=el('g',{class:'gvp'}, svg), gBands=el('g',{class:'gbands'}, vp), gEdges=el('g',{class:'gedges'}, vp), gNodes=el('g',{class:'gnodes'}, vp);
    var tip=document.createElement('div'); tip.className='gtip'; tip.hidden=true; stage.appendChild(tip);
    var legend=document.createElement('div'); legend.className='glegend';
    legend.innerHTML='<span><i class="lg-pos"></i>positive</span><span><i class="lg-neg"></i>negative</span><span><i class="lg-mix"></i>mixed</span><span><b class="lg-fn"></b>part (click to expand)</span><span><b class="lg-id"></b>identity</span><span><b class="lg-split"></b>split copy</span>';
    container.appendChild(legend);

    var T={k:1,x:0,y:0}, cur=null, nodeEls={}, lastP={}, firstDraw=true, edgeEls=[], mem=memLoad(opts.key), suppressUntil=0, nd=null;
    function stateKey(){ return Array.from(expanded).sort().join('|')+(showOff?'#off':''); }
    function belongs(p, id){ // a visible id (in some view) that sits inside part p
      if(id.slice(0,2)==='F:'){ var q=M.parts[id.slice(2)]; while(q){ if(q.name===p.name) return true; q=q.parent? M.parts[q.parent]:null; } return false; }
      return p.hidden.has(id);
    }
    function movedRec(id){ var st=mem.states[stateKey()]; if(st && st[id]) return st[id];
      if(id.slice(0,2)==='F:') return mem.fns[id.slice(2)] || null; if(id.slice(0,2)==='B:') return null; return mem.nodes[id] || null; }
    function remember(id){
      var q=cur.P[id]; if(!q) return; var sk=stateKey(); (mem.states[sk]=mem.states[sk]||{})[id]=[q.x,q.y];
      if(id.slice(0,2)==='F:') mem.fns[id.slice(2)]=[q.x,q.y]; else if(id.slice(0,2)!=='B:') mem.nodes[id]=[q.x,q.y];
      q.moved=true; memSave(opts.key, mem);
    }
    vp.style.transformOrigin='0 0';
    function apply(anim){ vp.style.transition=(anim && !REDUCED)? 'transform .28s ease' : 'none'; vp.style.transform='translate('+T.x+'px,'+T.y+'px) scale('+T.k+')'; }
    function fit(anim){
      if(!cur) return; var r=stage.getBoundingClientRect(); var W=r.width||600, H=r.height||400;
      var minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
      Object.keys(cur.P).forEach(function(id){ var p=cur.P[id]; minX=Math.min(minX,p.x-p.w/2-110); maxX=Math.max(maxX,p.x+p.w/2+40); minY=Math.min(minY,p.y-p.h/2-26); maxY=Math.max(maxY,p.y+p.h/2+16); });
      if(!isFinite(minX)) return;
      var k=Math.min(W/(maxX-minX), H/(maxY-minY), 1.4); k=Math.max(k, 0.04);
      T.k=k; T.x=(W-(maxX-minX)*k)/2-minX*k; T.y=(H-(maxY-minY)*k)/2-minY*k; apply(anim);
    }
    function setExpanded(s, fromSlider, noFit){ expanded=s; if(!fromSlider){ var L=0; for(var d=0; d<=M.maxDepth+1; d++){ var ok=M.partList.every(function(p){ return p.depth<d ? s.has(p.name) : true; }); if(ok) L=d; } slider.value=String(Math.min(L, M.maxDepth+1)); } draw(); if(!noFit) requestAnimationFrame(function(){ fit(true); }); }
    function members(p, V){ // visible ids belonging to part p (not its entries)
      var out=[]; V.order.forEach(function(id){ var v=V.nodes[id];
        if(v.kind==='function'){ var q=M.parts[v.part]; while(q){ if(q.name===p.name){ out.push(id); break; } q=q.parent?M.parts[q.parent]:null; } }
        else if(v.kind!=='bundle' && p.hidden.has(id)) out.push(id); });
      return out;
    }
    function showTip(html, ev){ tip.innerHTML=html; tip.hidden=false; var r=stage.getBoundingClientRect(); var x=ev.clientX-r.left+12, y=ev.clientY-r.top+12; tip.style.left=Math.min(x, r.width-260)+'px'; tip.style.top=Math.min(y, r.height-80)+'px'; }
    function esc(s){ return String(s==null?'':s).replace(/[&<>"]/g,function(c){ return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]; }); }
    function partOf(id){ var h=M.home[id]; return h? M.parts[h] : null; }
    function nodeTip(v){
      if(v.kind==='function'){ var p=M.parts[v.part]; return '<strong>'+esc(M.label(p))+'</strong><br><code>'+esc(p.name)+'</code> · '+p.hidden.size+' nodes inside, '+p.entries.size+' entries, '+p.exits.size+' exits'+(p.children.length?'<br>contains: '+p.children.map(function(c){ return esc(M.label(M.parts[c])); }).join(', '):'')+(p.reading?'<br><span class="gt-r">'+esc(p.reading)+'</span>':'')+'<br><em>click to expand</em>'; }
      if(v.kind==='bundle'){ return '<strong>'+v.members.length+' inputs</strong> into '+esc(M.label(M.parts[v.target.slice(2)]))+'<br>'+v.members.slice(0,40).map(function(id){ return esc(nodeLabel(M.nodes[id])); }).join(', ')+(v.members.length>40?' …':''); }
      var n=v.node, p=partOf(n.id);
      return '<strong>'+esc(nodeLabel(n))+'</strong> <code>'+esc(n.id)+'</code><br>'+esc(n.kind.replace('_',' '))+(n.act?' · '+esc(n.act):'')+(n.bias!=null?' · bias '+fmt(n.bias):'')+'<br>part: '+(p?esc(M.label(p))+' <code>'+esc(p.name)+'</code>':'none');
    }
    function edgeTip(e){
      var rows=e.list.slice(0,12).map(function(c){ return esc(nodeLabel(M.nodes[c.s]))+' → '+esc(nodeLabel(M.nodes[c.t]))+': '+fmt(c.w)+(c.on?'':' (disabled)'); });
      return (e.list.length>1? '<strong>'+e.list.length+' connections</strong><br>':'<strong>weight</strong> ')+rows.join('<br>')+(e.list.length>12?'<br>…':'');
    }

    function moveNode(id, x, y){
      var q=cur.P[id], g=nodeEls[id]; if(!q || !g) return;
      q.x=x; q.y=y; q.moved=true; g._target=q;
      g.style.transition='none'; g.setAttribute('transform','translate('+x+','+y+')');
      edgeEls.forEach(function(o){ if(o.e.s===id || o.e.t===id){ var a=cur.P[o.e.s], b=cur.P[o.e.t]; if(a&&b){ var d=edgePath(a,b); o.path.setAttribute('d', d); o.hit.setAttribute('d', d); } } });
      drawBands();
    }
    function drawBands(){
      var V=cur.V, P=cur.P, Lo=cur.Lo;
      gBands.replaceChildren();
      var bandList=M.partList.filter(function(p){ return expanded.has(p.name) && !(p.depth===0 && M.partList.filter(function(q){ return q.depth===0; }).length===1); });
      bandList.forEach(function(p){
        var ids=members(p, V); if(!ids.length) return;
        var minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity, anyMoved=false;
        ids.forEach(function(id){ var q=P[id]; if(!q) return; if(q.moved) anyMoved=true; minX=Math.min(minX,q.x-q.w/2); maxX=Math.max(maxX,q.x+q.w/2); minY=Math.min(minY,q.y-q.h/2); maxY=Math.max(maxY,q.y+q.h/2); });
        if(!isFinite(minX)) return;
        var gb=Lo.groups && Lo.groups[p.name]; var pad=12+4*(M.maxDepth-p.depth);
        if(gb && !anyMoved){ minY=gb.y0+14+pad; maxY=gb.y1-pad; }
        var g=el('g',{class:'gband'+(selected===p.name?' sel':''), 'data-part':p.name}, gBands);
        el('rect',{x:minX-pad, y:minY-pad-14, width:maxX-minX+2*pad, height:maxY-minY+2*pad+14, rx:12, fill:p.color, 'fill-opacity':0.13, stroke:p.color, 'stroke-opacity':0.8, 'stroke-width':1.5}, g);
        var t=el('text',{x:minX-pad+8, y:minY-pad-2, class:'gbandlbl', fill:p.color}, g); t.textContent='− '+M.label(p);
        g.addEventListener('click', function(ev){ ev.stopPropagation(); if(performance.now()<suppressUntil) return; var s=new Set(expanded); s.delete(p.name); selected=p.name; setExpanded(s); if(opts.onPart) opts.onPart(p.name); });
        g.addEventListener('mousemove', function(ev){ if(nd) return; showTip('<strong>'+esc(M.label(p))+'</strong> <code>'+esc(p.name)+'</code><br>click the label or band to collapse', ev); });
        g.addEventListener('mouseleave', function(){ tip.hidden=true; });
      });
    }
    function draw(){
      var V=view(M, expanded, showOff), Lo=layout(V,M), P=Lo.P; var prevP=lastP; cur={V:V,P:P,Lo:Lo};
      // moved positions: this collapse state's, else the node's last moved position; a function node with
      // moved members sits at the centroid of their current positions
      V.order.forEach(function(id){
        var r=movedRec(id);
        if(!r && id.slice(0,2)==='F:'){
          var part=M.parts[id.slice(2)], xs=0, ys=0, n=0, anyMoved=false;
          Object.keys(prevP).forEach(function(pid){ if(pid!==id && belongs(part, pid)){ xs+=prevP[pid].x; ys+=prevP[pid].y; n++; if(prevP[pid].moved) anyMoved=true; } });
          if(n && anyMoved) r=[xs/n, ys/n];
        }
        if(r){ P[id].x=r[0]; P[id].y=r[1]; P[id].moved=true; }
      });
      lvOut.textContent=slider.value+' / '+slider.max;
      // where should new nodes appear from? the old position of the function node that held them
      function origin(id){
        var v=V.nodes[id]; var nid=v.node? v.node.id : null; var q=nid? M.parts[M.home[nid]] : (v.kind==='function'? M.parts[v.part] : null);
        while(q){ if(prevP['F:'+q.name]) return prevP['F:'+q.name]; q=q.parent? M.parts[q.parent]:null; }
        return null;
      }
      drawBands();
      // edges
      gEdges.replaceChildren(); edgeEls=[];
      var maxW=0; V.edges.forEach(function(e){ maxW=Math.max(maxW, e.sum); });
      V.edges.forEach(function(e){
        var a=P[e.s], b=P[e.t]; if(!a||!b) return;
        var path=el('path',{d:edgePath(a,b), fill:'none', stroke:edgeColor(e), 'stroke-width':(0.6+3.6*Math.sqrt(e.sum/(maxW||1))).toFixed(2), 'stroke-opacity':e.off===e.list.length?0.5:0.8, class:'gedge'+(e.off===e.list.length?' off':'')}, gEdges);
        var hit=el('path',{d:edgePath(a,b), fill:'none', stroke:'transparent', 'stroke-width':8, class:'ghit'}, gEdges);
        hit.addEventListener('mousemove', function(ev){ path.classList.add('hov'); showTip(edgeTip(e), ev); });
        hit.addEventListener('mouseleave', function(){ path.classList.remove('hov'); tip.hidden=true; });
        edgeEls.push({e:e, path:path, hit:hit});
      });
      if(!REDUCED && !firstDraw){ gEdges.style.opacity='0'; gBands.style.opacity='0'; setTimeout(function(){ gEdges.style.opacity='1'; gBands.style.opacity='1'; }, 260); }
      // nodes (kept by id for animation)
      var keep={};
      V.order.forEach(function(id){
        var v=V.nodes[id], p=P[id]; keep[id]=1;
        var g=nodeEls[id], fresh=!g;
        if(fresh){
          g=el('g',{class:'gnode k-'+v.kind, tabindex:'0', role:'button', 'data-id':id}, gNodes);
          if(v.kind==='function'){
            var part=M.parts[v.part];
            el('rect',{class:'fbox', x:-p.w/2, y:-p.h/2, width:p.w, height:p.h, rx:10, fill:part.color, 'fill-opacity':0.28, stroke:part.color, 'stroke-width':2}, g);
            var tt=el('text',{class:'flbl', x:-p.w/2+10, y:4}, g); var lbl=M.label(part); tt.textContent=lbl.length>30? lbl.slice(0,29)+'…' : lbl;
            var pl=el('text',{class:'fplus', x:p.w/2-14, y:5}, g); pl.textContent=part.children.length? '+'+part.children.length : '+';
            g.setAttribute('aria-label','Part '+lbl+' (collapsed); Enter expands');
          } else if(v.kind==='bundle'){
            el('rect',{class:'bbox', x:-p.w/2, y:-p.h/2, width:p.w, height:p.h, rx:6}, g);
            var bt=el('text',{class:'blbl', x:0, y:4, 'text-anchor':'middle'}, g); bt.textContent=v.members.length+' inputs';
            g.setAttribute('aria-label', v.members.length+' inputs');
          } else {
            var n=v.node, k=v.kind;
            if(k==='identity'){ el('path',{class:'nshape', d:'M0,-8 L8,0 L0,8 L-8,0 Z'}, g); }
            else if(k==='input'||k==='split_input'){ el('rect',{class:'nshape'+(k==='split_input'?' split':''), x:-6, y:-6, width:12, height:12, rx:2}, g); }
            else { el('circle',{class:'nshape'+(k==='split'?' split':''), r:k==='output'?9:7}, g); if(k==='output') el('circle',{class:'nshape2', r:5}, g); }
            if(k==='input'||k==='split_input'){ var it=el('text',{class:'nlbl', x:-11, y:4, 'text-anchor':'end'}, g); it.textContent=nodeLabel(n); }
            else { var ht=el('text',{class:'nid', x:0, y:k==='output'?-14:-12, 'text-anchor':'middle'}, g); ht.textContent=k==='output'? (n.name||'output') : (n.name && n.name!==n.id ? n.name : n.id); }
            g.setAttribute('aria-label', (k.replace('_',' '))+' '+nodeLabel(n));
          }
          g.addEventListener('mousemove', function(ev){ showTip(nodeTip(v), ev); });
          g.addEventListener('mouseleave', function(){ tip.hidden=true; });
          g.addEventListener('click', function(ev){ ev.stopPropagation(); if(performance.now()<suppressUntil) return; activate(v, false); });
          g.addEventListener('keydown', function(ev){
            if(ev.key==='Enter'||ev.key===' '){ ev.preventDefault(); activate(v, true); return; }
            var d={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[ev.key];
            if(d){ ev.preventDefault(); var step=ev.shiftKey? 40 : 8, q=cur.P[id]; if(!q) return; moveNode(id, q.x+d[0]*step, q.y+d[1]*step); remember(id); }
          });
          g.addEventListener('pointerdown', function(ev){
            if(ev.button>0) return; ev.stopPropagation();
            var q=cur.P[id]; if(!q) return;
            nd={id:id, pid:ev.pointerId, x0:ev.clientX, y0:ev.clientY, px:q.x, py:q.y, moved:false};
            try{ g.setPointerCapture(ev.pointerId); }catch(e){}
          });
          g.addEventListener('pointermove', function(ev){
            if(!nd || nd.id!==id || nd.pid!==ev.pointerId) return;
            var dx=ev.clientX-nd.x0, dy=ev.clientY-nd.y0;
            if(!nd.moved && Math.abs(dx)+Math.abs(dy)<=4) return;
            if(!nd.moved){ nd.moved=true; g.classList.add('dragging'); }
            ev.stopPropagation(); tip.hidden=true;
            moveNode(id, nd.px+dx/T.k, nd.py+dy/T.k);
          });
          var endDrag=function(ev){
            if(!nd || nd.id!==id || nd.pid!==ev.pointerId) return;
            ev.stopPropagation();
            if(nd.moved){ suppressUntil=performance.now()+350; remember(id); g.classList.remove('dragging'); }
            try{ g.releasePointerCapture(ev.pointerId); }catch(e){}
            nd=null;
          };
          g.addEventListener('pointerup', endDrag); g.addEventListener('pointercancel', endDrag);
          g.addEventListener('focus', function(){ var r=g.getBoundingClientRect(), s=stage.getBoundingClientRect(); if(r.right<s.left||r.left>s.right||r.bottom<s.top||r.top>s.bottom){ var q=cur.P[id]; if(q){ T.x=s.width/2-q.x*T.k; T.y=s.height/2-q.y*T.k; apply(); } } });
          nodeEls[id]=g;
          var o=(!firstDraw && !REDUCED)? origin(id) : null;
          g.style.transition='none';
          g.setAttribute('transform','translate('+(o?o.x:p.x)+','+(o?o.y:p.y)+')');
          if(o){ g.style.opacity='0'; }
        }
        g.classList.toggle('sel', (v.kind==='function' && v.part===selected) || (v.node && selected && M.home[v.node.id]===selected));
        g._target=p;
      });
      Object.keys(nodeEls).forEach(function(id){ if(!keep[id]){ var g=nodeEls[id]; delete nodeEls[id]; g.remove(); } });
      // order DOM by visual order so Tab follows the layout
      V.order.slice().sort(function(a,b){ return P[a].x-P[b].x || P[a].y-P[b].y; }).forEach(function(id){ gNodes.appendChild(nodeEls[id]); });
      requestAnimationFrame(function(){ V.order.forEach(function(id){ var g=nodeEls[id]; if(!g) return; var p=g._target; g.style.transition=(REDUCED||firstDraw)?'none':''; g.setAttribute('transform','translate('+p.x+','+p.y+')'); g.style.opacity='1'; }); });
      lastP=P;
      if(firstDraw){ firstDraw=false; requestAnimationFrame(function(){ fit(false); }); }
      if(opts.onDraw) opts.onDraw(V);
    }
    function activate(v, key){
      if(v.kind==='function'){ var s=new Set(expanded); s.add(v.part); selected=v.part; setExpanded(s); if(opts.onPart) opts.onPart(v.part); return; }
      if(v.kind==='bundle'){ return; }
      var h=M.home[v.node.id];
      if(key && h){ // Enter on a node inside an expanded part collapses that part
        var p=M.parts[h]; while(p && !expanded.has(p.name)) p=p.parent? M.parts[p.parent]:null;
        if(p && p.depth>0){ var s2=new Set(expanded); s2.delete(p.name); selected=p.name; setExpanded(s2); if(opts.onPart) opts.onPart(p.name); return; }
      }
      selected=h; draw(); if(h && opts.onPart) opts.onPart(h);
    }
    // pan / zoom
    var pointers={}, dragged=false, start=null;
    svg.addEventListener('wheel', function(ev){ ev.preventDefault(); var r=svg.getBoundingClientRect(); var mx=ev.clientX-r.left, my=ev.clientY-r.top; var f=Math.exp(-ev.deltaY*0.0015); var k2=Math.min(6, Math.max(0.03, T.k*f)); f=k2/T.k; T.x=mx-(mx-T.x)*f; T.y=my-(my-T.y)*f; T.k=k2; apply(); }, {passive:false});
    svg.addEventListener('pointerdown', function(ev){ pointers[ev.pointerId]={x:ev.clientX,y:ev.clientY}; dragged=false; start={x:ev.clientX,y:ev.clientY,T:{k:T.k,x:T.x,y:T.y}}; if(Object.keys(pointers).length===2){ var p=Object.values(pointers); start.d=Math.hypot(p[0].x-p[1].x,p[0].y-p[1].y); start.m={x:(p[0].x+p[1].x)/2, y:(p[0].y+p[1].y)/2}; } });
    svg.addEventListener('pointermove', function(ev){
      if(!pointers[ev.pointerId]) return; pointers[ev.pointerId]={x:ev.clientX,y:ev.clientY};
      var ps=Object.values(pointers);
      if(ps.length===1 && start){ var dx=ev.clientX-start.x, dy=ev.clientY-start.y; if(Math.abs(dx)+Math.abs(dy)>4){ dragged=true; try{ svg.setPointerCapture(ev.pointerId); }catch(e){} } if(dragged){ T.x=start.T.x+dx; T.y=start.T.y+dy; apply(); tip.hidden=true; } }
      else if(ps.length===2 && start && start.d){ dragged=true; var d=Math.hypot(ps[0].x-ps[1].x, ps[0].y-ps[1].y); var r=svg.getBoundingClientRect(); var mx=start.m.x-r.left, my=start.m.y-r.top; var k2=Math.min(6, Math.max(0.03, start.T.k*d/start.d)); var f=k2/start.T.k; T.k=k2; T.x=mx-(mx-start.T.x)*f; T.y=my-(my-start.T.y)*f; apply(); }
    });
    function up(ev){ delete pointers[ev.pointerId]; if(!Object.keys(pointers).length){ setTimeout(function(){ dragged=false; }, 0); start=null; } }
    svg.addEventListener('pointerup', up); svg.addEventListener('pointercancel', up);
    svg.addEventListener('click', function(){ if(!dragged && performance.now()>=suppressUntil){ selected=null; draw(); if(opts.onPart) opts.onPart(null, true); } });
    draw();
    return {
      select: function(name, expand){
        if(!M.parts[name]) return;
        var s=new Set(expanded), q=M.parts[name].parent? M.parts[M.parts[name].parent] : null;
        while(q){ s.add(q.name); q=q.parent? M.parts[q.parent]:null; }
        if(expand) s.add(name);
        selected=name; setExpanded(s, false, true);
        var V=cur.V, P=cur.P, ids=members(M.parts[name], V).concat(V.nodes['F:'+name]?['F:'+name]:[]);
        if(ids.length){ var r=stage.getBoundingClientRect(); var x=0,y=0; ids.forEach(function(id){ x+=P[id].x; y+=P[id].y; }); x/=ids.length; y/=ids.length; var sx=T.x+x*T.k, sy=T.y+y*T.k; if(sx<0||sx>r.width||sy<0||sy>r.height){ T.x=r.width/2-x*T.k; T.y=r.height/2-y*T.k; apply(); } }
      },
      state: function(){ var V=cur.V; return {visible:V.order.length, functions:V.order.filter(function(id){ return V.nodes[id].kind==='function'; }).map(function(id){ return V.nodes[id].part; }), bands:gBands.childNodes.length, nodes:Object.keys(M.nodes).length, edges:V.edges.length}; },
      fit: function(){ fit(false); },
      pos: function(id){ var q=cur.P[id]; return q? {x:q.x, y:q.y, moved:!!q.moved} : null; },
      view: function(){ return {k:T.k, x:T.x, y:T.y}; }
    };
  }
  return {mount:mount, thumb:thumb, prepare:prepare};
})();
