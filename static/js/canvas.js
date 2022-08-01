window.addEventListener('load', () => {
    const redraw = document.getElementById("redraw")
    const predict = document.getElementById("predict")
    const canvas = document.querySelector('#canvas');
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    //变量
    let drawing = false
    offset_x = canvas.offsetLeft;
    offset_y = canvas.offsetTop;
    let xhr = new XMLHttpRequest();

   //方法
    function startPosition(e) {
        drawing = true;
        draw(e)
    }
    
    function endPosition() {
        drawing = false;
        ctx.beginPath();
    }
    
    function draw(e) {
        if (!drawing) return;
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'white';
        
        ctx.lineTo(e.clientX - offset_x, e.clientY - offset_y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(e.clientX - offset_x, e.clientY - offset_y)
    }

    function resetCanvas() {
        const canvas = document.querySelector('#canvas');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    function getCanvasData() {
        let canvas = document.getElementById("canvas");
        let dataURL = canvas.toDataURL("image/png");
        document.getElementById('data').value = dataURL;
        let fd = new FormData(document.forms["hidden_form"]);

        let xhr = new XMLHttpRequest({mozSystem: true});
        xhr.open('POST', '/predict', true);

        xhr.onreadystatechange = function () {
            if (xhr.readyState == XMLHttpRequest.DONE) {
                document.getElementById('result').innerHTML = xhr.responseText;

            }
        }

        xhr.onload = function() {

        };
        xhr.send(fd);
    };
    
    //事件监听
    canvas.addEventListener('mousedown', startPosition);
    canvas.addEventListener('mouseup', endPosition);
    canvas.addEventListener('mousemove', draw)
    redraw.addEventListener('click', resetCanvas)
    predict.addEventListener('click', getCanvasData)
})