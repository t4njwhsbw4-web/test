(function(){
var WA='https://wa.me/41766093000',TEL='tel:+41766093000';
var SVGW='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12.04 2C6.58 2 2.13 6.45 2.13 11.91c0 1.75.46 3.45 1.32 4.95L2 22l5.25-1.38c1.45.79 3.08 1.21 4.79 1.21h.01c5.46 0 9.91-4.45 9.91-9.91C21.96 6.45 17.5 2 12.04 2m0 18.15h-.01c-1.52 0-3.01-.41-4.3-1.18l-.31-.18-3.19.84.85-3.11-.2-.32a8.2 8.2 0 0 1-1.26-4.39c0-4.54 3.7-8.23 8.24-8.23 2.2 0 4.27.86 5.83 2.42a8.18 8.18 0 0 1 2.41 5.82c0 4.54-3.69 8.23-8.26 8.23m4.52-6.16c-.25-.12-1.47-.72-1.69-.81-.23-.08-.39-.12-.56.13-.16.24-.64.8-.78.97-.14.16-.29.18-.54.06-.25-.13-1.05-.39-1.99-1.23-.74-.66-1.23-1.47-1.38-1.72-.14-.25-.01-.38.11-.5.11-.11.25-.29.37-.43.13-.15.17-.25.25-.41.08-.17.04-.31-.02-.43-.06-.12-.56-1.34-.76-1.84-.2-.48-.41-.42-.56-.43h-.48c-.17 0-.43.06-.66.31-.23.25-.86.85-.86 2.07s.89 2.4 1.01 2.56c.12.17 1.74 2.66 4.22 3.73.59.25 1.05.4 1.41.52.59.19 1.13.16 1.56.1.47-.07 1.47-.6 1.67-1.18.21-.58.21-1.07.15-1.18-.06-.11-.22-.17-.47-.29"/></svg>';
var SVGT='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6.6 10.8a15.1 15.1 0 0 0 6.6 6.6l2.2-2.2c.3-.3.7-.4 1-.2 1.1.4 2.3.6 3.6.6.6 0 1 .4 1 1V20c0 .6-.4 1-1 1A17 17 0 0 1 3 4c0-.6.4-1 1-1h3.4c.6 0 1 .4 1 1 0 1.2.2 2.4.6 3.6.1.3 0 .7-.2 1z"/></svg>';
var CSS=''
+'.nvq{margin-top:34px;border:1px solid var(--grey-2);border-radius:6px;padding:34px 34px 30px;background:var(--white)}'
+'.nvq-top{display:flex;align-items:baseline;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:6px}'
+'.nvq-step{font-family:var(--display);font-weight:700;font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--grey-3);opacity:.7}'
+'.nvq h3{font-size:22px;margin:0}'
+'.nvq-sub{color:var(--grey-3);font-size:15.5px;margin:8px 0 0}'
+'.nvq-opts{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin-top:22px}'
+'.nvq-opt{text-align:left;background:var(--white);border:1px solid var(--grey-2);border-radius:4px;padding:15px 17px;font-family:var(--text);font-size:15.5px;color:var(--ink);cursor:pointer;transition:border-color .2s,background .2s}'
+'.nvq-opt:hover{border-color:var(--ink)}'
+'.nvq-opt b{display:block;font-family:var(--display);font-weight:700;font-size:16px;margin-bottom:2px}'
+'.nvq-opt span{color:var(--grey-3);font-size:13.5px}'
+'.nvq-foot{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin-top:22px}'
+'.nvq-back{background:none;border:0;padding:0;font-family:var(--display);font-weight:700;font-size:14.5px;color:var(--grey-3);cursor:pointer}'
+'.nvq-back:hover{color:var(--ink)}'
+'.nvq-res{display:flex;align-items:flex-end;gap:14px;flex-wrap:wrap;margin-top:18px}'
+'.nvq-price{font-family:var(--display);font-weight:700;font-size:40px;letter-spacing:-0.02em;line-height:1}'
+'.nvq-unit{color:var(--grey-3);font-size:15px;padding-bottom:4px}'
+'.nvq-sum{margin-top:16px;color:var(--grey-3);font-size:15px;line-height:1.7}'
+'.nvq-sum b{color:var(--ink);font-weight:700}'
+'@media(max-width:620px){.nvq{padding:26px 22px 24px}.nvq-price{font-size:34px}}'
+'.cta-contactrow{display:flex;align-items:center;gap:12px;margin-top:12px;flex-wrap:wrap}'
+'.cta-wa{display:inline-flex;align-items:center;justify-content:center;width:42px;height:42px;flex:none;border-radius:4px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.16);transition:background .2s,border-color .2s}'
+'.cta-wa svg{width:20px;height:20px;fill:rgba(255,255,255,.85)}'
+'.cta-wa:hover{background:#25D366;border-color:#25D366}'
+'.cta-wa:hover svg{fill:#fff}'
+'.cta-contactrow .cta-tel{margin-top:0}'
+'.nvbar{position:fixed;left:0;right:0;bottom:0;z-index:150;background:rgba(26,26,26,.96);backdrop-filter:blur(10px);border-top:1px solid rgba(255,255,255,.1);transform:translateY(110%);transition:transform .3s ease}'
+'.nvbar.in{transform:none}'
+'.nvbar-in{max-width:1180px;margin:0 auto;padding:11px 24px;display:flex;align-items:center;justify-content:space-between;gap:16px}'
+'.nvbar-txt{color:rgba(255,255,255,.72);font-size:14.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}'
+'.nvbar-txt b{color:#fff;font-family:var(--display);font-weight:700}'
+'.nvbar-ac{display:flex;align-items:center;gap:10px;flex:none}'
+'.nvbar-ico{display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;border-radius:4px;border:1px solid rgba(255,255,255,.18);transition:background .2s,border-color .2s}'
+'.nvbar-ico svg{width:19px;height:19px;fill:rgba(255,255,255,.85)}'
+'.nvbar-ico:hover{background:rgba(255,255,255,.1);border-color:rgba(255,255,255,.4)}'
+'.nvbar-wa:hover{background:#25D366;border-color:#25D366}'
+'.nvbar-wa:hover svg{fill:#fff}'
+'.nvbar-btn{background:var(--pink);color:var(--ink);font-family:var(--display);font-weight:700;font-size:15px;padding:11px 26px;border:0;border-radius:4px;cursor:pointer;white-space:nowrap;transition:box-shadow .2s}'
+'.nvbar-btn:hover{box-shadow:0 8px 22px rgba(247,146,208,.35)}'
+'#nv-host{padding-bottom:64px}'
+'@media(max-width:620px){.nvbar-in{padding:9px 16px;gap:10px}.nvbar-txt{display:none}.nvbar-ac{flex:1;justify-content:space-between}.nvbar-btn{flex:1;text-align:center}}'
+'@media(min-width:721px){.nvbar-tel{display:none}}';
var FLOW={
einzel:{label:'Einzelne Videos',q:'Wie viele Videos brauchen Sie?',opts:[
{k:'1',t:'Ein Video',s:'einmalig',price:250,unit:'einmalig',txt:'1 Video'},
{k:'2',t:'Zwei Videos',s:'aus einem Drehtermin',price:500,unit:'einmalig',txt:'2 Videos'},
{k:'3',t:'Drei oder mehr',s:'CHF 200 pro Video',price:600,from:1,unit:'für 3 Videos, einmalig',txt:'3 oder mehr Videos'}]},
film:{label:'Unternehmensfilm',q:'Wie umfangreich soll der Film werden?',opts:[
{k:'s',t:'60 bis 90 Sekunden',s:'der übliche Umfang',price:890,unit:'einmalig',txt:'60–90 Sekunden'},
{k:'l',t:'Länger oder aufwendiger',s:'mehrere Standorte, Interviews',price:890,from:1,unit:'nach Offerte',txt:'länger oder aufwendiger'}]},
partner:{label:'Content-Partnerschaft',q:'Wie viele Videos pro Monat?',opts:[
{k:'3',t:'Drei pro Monat',s:'der Standardumfang',price:590,from:1,unit:'pro Monat',txt:'3 Videos pro Monat'},
{k:'m',t:'Mehr als drei',s:'wir rechnen es gemeinsam aus',price:590,from:1,unit:'pro Monat, nach Offerte',txt:'mehr als 3 Videos pro Monat'}]},
immo:{label:'Immobilien-Objektvideo',q:'Wie viele Objekte?',opts:[
{k:'1',t:'Ein Objekt',s:'Wohnung, Haus oder Gewerbe',price:550,from:1,unit:'einmalig',txt:'1 Objekt'},
{k:'m',t:'Mehrere Objekte',s:'regelmässig oder als Paket',price:550,from:1,unit:'pro Objekt',txt:'mehrere Objekte'}]},
offen:{label:'Noch offen',q:'Worum geht es ungefähr?',opts:[
{k:'b',t:'Ich möchte beraten werden',s:'wir schauen es gemeinsam an',price:0,unit:'',txt:'Beratung gewünscht'},
{k:'p',t:'Nur eine Preisauskunft',s:'unverbindlich',price:0,unit:'',txt:'Preisauskunft'}]}
};
var WHEN=[{k:'a',t:'So bald wie möglich',s:'innert vier Wochen',txt:'so bald wie möglich'},
{k:'b',t:'In ein bis drei Monaten',s:'in Planung',txt:'in 1–3 Monaten'},
{k:'c',t:'Noch offen',s:'erst einmal informieren',txt:'Zeitpunkt noch offen'}];
var ST={a:null,b:null,c:null};
function el(tag,cls,html){var e=document.createElement(tag);if(cls)e.className=cls;if(html!=null)e.innerHTML=html;return e}
function money(n){return 'CHF '+String(n).replace(/\B(?=(\d{3})+(?!\d))/g,'’')}
function render(){
var box=document.getElementById('nv-rechner');if(!box)return;
var step=ST.a?(ST.b?3:2):1;
var head,sub,opts=[],back=null;
if(step===1){head='Was brauchen Sie?';sub='Ein Klick genügt – Sie sehen sofort den Richtpreis.';
opts=[{k:'einzel',t:'Einzelne Videos',s:'kurze Videos für Website und Kanäle'},
{k:'film',t:'Unternehmensfilm',s:'ein Film über Ihre Firma'},
{k:'partner',t:'Content-Partnerschaft',s:'regelmässig Videos pro Monat'},
{k:'immo',t:'Immobilien-Objektvideo',s:'Wohnung, Haus oder Gewerbe'},
{k:'offen',t:'Weiss ich noch nicht',s:'wir beraten Sie'}];
}else if(step===2){var f=FLOW[ST.a];head=f.q;sub='';opts=f.opts;back=1}
else{head='Wann soll es losgehen?';sub='';opts=WHEN;back=2}
box.innerHTML='';
var top=el('div','nvq-top');
top.appendChild(el('span','nvq-step','Schritt '+step+' von 3'));
box.appendChild(top);
box.appendChild(el('h3',null,head));
if(sub)box.appendChild(el('p','nvq-sub',sub));
var grid=el('div','nvq-opts');
opts.forEach(function(o){
var b=el('button','nvq-opt','<b>'+o.t+'</b><span>'+(o.s||'')+'</span>');
b.type='button';
b.addEventListener('click',function(){
if(step===1){ST.a=o.k;ST.b=null;ST.c=null}
else if(step===2){ST.b=o}
else{ST.c=o;return result()}
render();
});
grid.appendChild(b);
});
box.appendChild(grid);
if(back){
var fo=el('div','nvq-foot');
var bb=el('button','nvq-back','← Zurück');bb.type='button';
bb.addEventListener('click',function(){if(back===1){ST.a=null;ST.b=null}else{ST.b=null}render()});
fo.appendChild(bb);box.appendChild(fo);
}
}
function summary(){
var f=FLOW[ST.a];
return 'Angebot: '+f.label+'\nUmfang: '+ST.b.txt+'\nZeitpunkt: '+ST.c.txt+(ST.b.price?'\nRichtpreis: '+(ST.b.from?'ab ':'')+money(ST.b.price)+(ST.b.unit?' ('+ST.b.unit+')':''):'');
}
function result(){
var box=document.getElementById('nv-rechner');if(!box)return;
var f=FLOW[ST.a],o=ST.b;
box.innerHTML='';
var top=el('div','nvq-top');
top.appendChild(el('span','nvq-step','Ihr Richtpreis'));
box.appendChild(top);
box.appendChild(el('h3',null,f.label));
if(o.price){
var r=el('div','nvq-res');
r.appendChild(el('div','nvq-price',(o.from?'ab ':'')+money(o.price)));
if(o.unit)r.appendChild(el('span','nvq-unit',o.unit));
box.appendChild(r);
}else{
box.appendChild(el('p','nvq-sub','Wir schauen uns Ihr Vorhaben an und nennen Ihnen einen konkreten Preis – unverbindlich.'));
}
box.appendChild(el('p','nvq-sum','<b>'+o.t+'</b> · '+ST.c.t+'<br>Richtpreis für den beschriebenen Umfang. Massgebend ist die schriftliche Offerte.'));
var fo=el('div','nvq-foot');
var go=el('button','btn btn-pink','Anfrage übernehmen');go.type='button';
go.addEventListener('click',function(){
var form=document.getElementById('nv-form');
var ta=form&&form.elements['nachricht'];
if(ta){ta.value='Anfrage über den Preisrechner:\n\n'+summary()+'\n\n';}
var k=document.getElementById('kontakt');
if(k)k.scrollIntoView({behavior:'smooth',block:'start'});
setTimeout(function(){try{var n=form&&form.elements['name'];if(n)n.focus({preventScroll:true})}catch(e){}},700);
});
var again=el('button','nvq-back','Neu starten');again.type='button';
again.addEventListener('click',function(){ST.a=null;ST.b=null;ST.c=null;render()});
fo.appendChild(go);fo.appendChild(again);
box.appendChild(fo);
var bar=document.getElementById('nv-bar-txt');
if(bar&&o.price)bar.innerHTML='Ihr Richtpreis: <b>'+(o.from?'ab ':'')+money(o.price)+'</b>'+(o.unit?' · '+o.unit:'');
}
function mountQuiz(h){
if(h.querySelector('#nv-rechner'))return;
var note=h.querySelector('#preise .pricing-note');if(!note)return;
var box=el('div','nvq');box.id='nv-rechner';
note.parentNode.insertBefore(box,note);
render();
}
function mountWa(h){
var tel=h.querySelector('.cta-direct .cta-tel');
if(!tel||tel.parentNode.classList.contains('cta-contactrow'))return;
var row=el('div','cta-contactrow');
var wa=el('a','cta-wa',SVGW);
wa.href=WA;wa.target='_blank';wa.rel='noopener';wa.setAttribute('aria-label','WhatsApp an Novara Media');
tel.parentNode.insertBefore(row,tel);
row.appendChild(wa);row.appendChild(tel);
}
function mountBar(h){
if(document.getElementById('nv-bar'))return;
var bar=el('div','nvbar');bar.id='nv-bar';
var inner=el('div','nvbar-in');
var txt=el('span','nvbar-txt','Videos <b>ab CHF 250</b> · Antwort innert 24 Stunden');
txt.id='nv-bar-txt';
var ac=el('div','nvbar-ac');
var wa=el('a','nvbar-ico nvbar-wa',SVGW);wa.href=WA;wa.target='_blank';wa.rel='noopener';wa.setAttribute('aria-label','WhatsApp');
var te=el('a','nvbar-ico nvbar-tel',SVGT);te.href=TEL;te.setAttribute('aria-label','Anrufen');
var btn=el('button','nvbar-btn','Anfragen');btn.type='button';
btn.addEventListener('click',function(){var k=document.getElementById('kontakt');if(k)k.scrollIntoView({behavior:'smooth',block:'start'})});
ac.appendChild(wa);ac.appendChild(te);ac.appendChild(btn);
inner.appendChild(txt);inner.appendChild(ac);bar.appendChild(inner);
h.appendChild(bar);
var k=document.getElementById('kontakt');
function upd(){
var y=window.pageYOffset||document.documentElement.scrollTop;
var hide=false;
if(k){var r=k.getBoundingClientRect();if(r.top<window.innerHeight*0.75)hide=true}
if(y>620&&!hide)bar.classList.add('in');else bar.classList.remove('in');
}
window.addEventListener('scroll',upd,{passive:true});
window.addEventListener('resize',upd);
upd();
}
function install(){
var h=document.getElementById('nv-host');
if(!h)return;
if(h.getAttribute('data-nvx')==='1')return;
h.setAttribute('data-nvx','1');
if(!document.getElementById('nv-x-css')){
var s=document.createElement('style');s.id='nv-x-css';s.textContent=CSS;document.head.appendChild(s);
}
ST.a=null;ST.b=null;ST.c=null;
mountQuiz(h);mountWa(h);mountBar(h);
}
install();
setInterval(install,500);
})();
