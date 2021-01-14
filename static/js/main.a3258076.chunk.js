(this.webpackJsonptflowjstest=this.webpackJsonptflowjstest||[]).push([[0],{358:function(e,t){},359:function(e,t){},367:function(e,t){},370:function(e,t){},371:function(e,t){},449:function(e,t,n){"use strict";n.r(t);var a=n(55),r=n(41),c=n.n(r),i=n(315),s=n.n(i),o=n(135),u=n(2),l=n.n(u),d=n(15),h=n(6),b=n(104),j=n(237);function f(e){return p.apply(this,arguments)}function p(){return(p=Object(d.a)(l.a.mark((function e(t){var n,a,r,c,i,s,o,u,d;return l.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return n=t.batchSize,a=t.learningRate,r=t.epochs,"datasets/iris.csv",c=b.data.csv("datasets/iris.csv",{columnConfigs:{species:{isLabel:!0}}}),e.next=5,c.columnNames();case 5:return e.t0=e.sent.length,i=e.t0-1,s=c.map((function(e){var t=e.xs,n=e.ys,a=["setosa"===n.species?1:0,"virginica"===n.species?1:0,"versicolor"===n.species?1:0];return{xs:Object.values(t),ys:Object.values(a)}})).batch(n),(o=b.sequential()).add(b.layers.dense({inputShape:[i],activation:"sigmoid",units:5})),o.add(b.layers.dense({activation:"softmax",units:3})),o.compile({optimizer:b.train.adam(a),loss:"categoricalCrossentropy"}),u={name:"Loss",tab:"Training"},d=[],e.next=16,o.fitDataset(s,{epochs:r,callbacks:{onTrainEnd:function(){return alert("Training is done. Now you can predict a value")},onEpochEnd:function(e,t){d.push(t),j.show.history(u,d,["loss"]),console.log("Epoch: ".concat(e," | Loss: ").concat(t.loss))}}});case 16:return e.abrupt("return",o);case 17:case"end":return e.stop()}}),e)})))).apply(this,arguments)}var x=n(107),v=n(108);function g(){var e=Object(x.a)(["\n  width: 150px;\n  height: 50px;\n  text-align: center;\n  font-size: 0.9rem;\n"]);return g=function(){return e},e}function O(){var e=Object(x.a)(["\n  width: 100%;\n  height: 25px;\n  border-radius: 7.5px;\n  border: 1px solid #e6e6e6;\n  box-sizing: border-box;\n"]);return O=function(){return e},e}var m=v.a.input(O()),y=v.a.label(g());function w(){var e=Object(x.a)(["\n  height: 300px;\n  display: flex;\n  flex-direction: column;\n  justify-content: space-evenly;\n"]);return w=function(){return e},e}var S=v.a.div(w());function C(){var e=Object(x.a)(["\n  min-width: 100px;\n  height: 35px;\n  margin: 10px 15px;\n  border: none;\n  border-radius: 7.5px;\n  font-weight: 600;\n  background: rgb(0, 122, 255);\n  color: white;\n"]);return C=function(){return e},e}var k=v.a.button(C());function z(){var e=Object(x.a)(["\n  font-size: 32px;\n"]);return z=function(){return e},e}var V=v.a.h1(z());function E(){var e=Object(r.useState)(null),t=Object(h.a)(e,2),n=t[0],c=t[1],i=Object(r.useState)([4.6,3.4,1.4,.3]),s=Object(h.a)(i,2),u=s[0],p=s[1],x=Object(r.useState)({batchSize:10,learningRate:.05,epochs:20}),v=Object(h.a)(x,2),g=v[0],O=v[1],w=function(){var e=Object(d.a)(l.a.mark((function e(){var t;return l.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.next=2,f(g);case 2:t=e.sent,c(t);case 4:case"end":return e.stop()}}),e)})));return function(){return e.apply(this,arguments)}}();return Object(a.jsxs)("div",{children:[Object(a.jsx)(V,{children:"Iris Classification"}),Object(a.jsx)("a",{href:"https://archive.ics.uci.edu/ml/datasets/iris",children:"dataset"}),Object(a.jsxs)(S,{children:[Object(a.jsxs)(y,{children:["Batch size:",Object(a.jsx)(m,{onChange:function(e){return O(Object(o.a)(Object(o.a)({},g),{},{batchSize:+e.target.value}))},defaultValue:g.batchSize})]}),Object(a.jsxs)(y,{children:["Learning rate:",Object(a.jsx)(m,{onChange:function(e){return O(Object(o.a)(Object(o.a)({},g),{},{learningRate:+e.target.value}))},defaultValue:g.learningRate})]}),Object(a.jsxs)(y,{children:["Epochs:",Object(a.jsx)(m,{onChange:function(e){return O(Object(o.a)(Object(o.a)({},g),{},{epochs:+e.target.value}))},defaultValue:g.epochs})]}),Object(a.jsxs)(y,{children:["Value to predict:",Object(a.jsx)(m,{onChange:function(e){return p(JSON.parse(e.target.value))},defaultValue:JSON.stringify(u)})]})]}),Object(a.jsx)(k,{onClick:w,children:"Train"}),Object(a.jsx)(k,{onClick:function(){return function(e,t){if(4===t.length)if(e){var n=b.tensor2d(t,[1,4]),a=b.argMax(e.predict(n),1).dataSync();alert(["Setosa","Virginica","Versicolor"][a])}else alert("First you need to train a model");else alert("Tensor should have 4 values")}(n,u)},children:"Predict"}),Object(a.jsx)(k,{onClick:function(){return j.visor().toggle()},children:"Toggle visor"})]})}s.a.render(Object(a.jsx)(c.a.StrictMode,{children:Object(a.jsx)(E,{})}),document.getElementById("root"))}},[[449,1,2]]]);
//# sourceMappingURL=main.a3258076.chunk.js.map