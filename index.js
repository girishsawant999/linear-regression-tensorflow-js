let x_vals = [];
let y_vals = [];

let m, b;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

const slope = document.querySelector("#slope");
const yIntercept = document.querySelector("#yintercept");

function setup() {
  createCanvas(1000, 1000);
  background(0);
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

function predict(x_vals) {
  const xs = tf.tensor1d(x_vals);
  // y = mx + b
  const ys = xs.mul(m).add(b);
  tf.dispose(xs);
  return ys;
}

function loss(pred, label) {
  return pred.sub(label).square().mean();
}

function mousePressed() {
  const x = map(mouseX, 0, width, 0, 1);
  const y = map(mouseY, 0, height, 1, 0);
  x_vals.push(x);
  y_vals.push(y);
}

function getRandomArbitrary(min = 0, max = 255) {
  return Math.random() * (max - min) + min;
}

function draw() {
  tf.tidy(() => {
    if (x_vals.length > 0) {
      const ys = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x_vals), ys));
    }
  });

  background(0);
  stroke(255);
  fill(getRandomArbitrary());
  strokeWeight(5);
  for (i = 0; i < x_vals.length; i++) {
    const px = map(x_vals[i], 0, 1, 0, width);
    const py = map(y_vals[i], 0, 1, height, 0);
    point(px, py);
  }

  let _m = m.dataSync();
  let _b = b.dataSync();
  slope.innerHTML = `Slope = ${_m[0].toFixed(2)}`;
  yIntercept.innerHTML = `Y-intercept = ${_b[0].toFixed(2)}`;

  const xs = [0, 1];
  let ys = tf.tidy(() => predict(xs));
  _ys = ys.dataSync();
  tf.dispose(ys);

  let x0 = map(xs[0], 0, 1, 0, width);
  let x1 = map(xs[1], 0, 1, 0, width);
  let y0 = map(_ys[0], 0, 1, height, 0);
  let y1 = map(_ys[1], 0, 1, height, 0);
  strokeWeight(2);
  line(x0, y0, x1, y1);
  console.log("numTensors :>> ", tf.memory().numTensors);
}
