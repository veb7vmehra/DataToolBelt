(function ($) {
    $.fn.fileUplodPlugins = function (option) {
        let allData = []
        let settings = $.extend({
            inputFileUpload: "#fileId",
            ValidType: ['image/jpeg', 'image/png', ],
            btnUpload: ".file-upload-wrapper-title__btn",
            deleteImageBtn: ".image-previwe__delete-btn",
            boxFileUploadPreviwe: '.image-previwe',
            boxErrorPreviwe: '.error-wrapper',
        }, option)
        

        function removeElement(id) {

            allData = allData.filter(itme => {
                return itme.id != id
            })
        }
        $(document).ready(function () {
            const removeBtn = (id) => {
                let span = document.createElement('span');
                span.setAttribute('class', 'image-previwe__delete-btn');
                span.setAttribute('data-id', id);
                return span;
            }
            const previweNameImage = (name) => {
                let div = document.createElement('div');
                div.setAttribute('class', "image-previwe__hover");
                let p = document.createElement('p');
                p.innerHTML = name;
                div.appendChild(p)
                return div;
            }
            const CreatImagePreview = (image, id, noImage = null) => {
                let div = document.createElement('div');
                div.setAttribute("class", "image-previwe__image");
                if (noImage != null) {
                    div.setAttribute("style", `background-image:url(${noImage})`);
                } else {
                    div.setAttribute("style", `background-image:url(${image})`);
                }
                div.appendChild(removeBtn(id));
                div.appendChild(previweNameImage(id));
                return div;
            }
            const createError = () => {
                let div = document.createElement('div');
                div.setAttribute("class", 'error-format');
                div.innerHTML = "File Format Is Not Valid";
                return div;
            }
            let checkTypeFiles = (file) => {
                let validation = {
                    isValid: false,
                    preViwe: null,
                }

                settings.ValidType.forEach((data, index) => {

                    if (file.type == data) {
                        file.previewimg = null;
                        validation.isValid = true;
                        validation.preViwe = file;
                        switch (true) {
                            case file.type == 'application/pdf':
                                file.previewimg = 'assets/img/pdf-icon.png'
                                validation.preViwe = file;
                                break;
                            case file.type == 'text/plain':
                                file.previewimg = 'assets/img/txt-icon.png';
                                validation.preViwe = file;
                                break;
                        }
                    }
                })

                return validation;
            }

            function uploadClick(input) {
                let inputValue = Object.values(input.files)
                if (input.files.length) {
                    inputValue.forEach(element => {
                        let isElementVaild = checkTypeFiles(element)

                        if (isElementVaild.isValid) {
                            allData.push({
                                id: element.lastModified,
                                file: element
                            });
                            renderImageData(isElementVaild.preViwe);
                        } else {
                            $(settings.boxErrorPreviwe).append(createError());
                            setTimeout(() => {
                                $('.error-format').fadeOut("slow")
                            }, 2500);
                        }
                    });
                }
            }

            function DragAndDropUpload(input) {
                let inputValue = Object.values(input)
                if (input.length) {
                    inputValue.forEach((element) => {
                        let isElementVaild = checkTypeFiles(element)

                        if (isElementVaild.isValid) {
                            allData.push({
                                id: element.lastModified,
                                file: element
                            });
                            renderImageData(isElementVaild.preViwe);
                        } else {
                            settings.boxErrorPreviwe.appendChild(createError());
                        }
                    });
                }
            }

            function renderImageData(arrayItme) {
                var reader = new FileReader();

                reader.onload = function (e) {
                    let previweTag = CreatImagePreview(e.target.result, arrayItme.lastModified, arrayItme.previewimg);
                    $(settings.boxFileUploadPreviwe).append(previweTag);
                }

                reader.readAsDataURL(arrayItme);
            };
            $(settings.inputFileUpload).change(function () {
                uploadClick(this);
            });

            $("html").on("dragover", function (e) {
                e.preventDefault();
                e.stopPropagation();
            });
            $('.box-fileupload').on('dragenter', function (e) {
                e.stopPropagation();
                e.preventDefault();
                $(".box-fileupload__lable").text("Drop");
            });
            $('.box-fileupload').on('drop', function (e) {
                e.stopPropagation();
                e.preventDefault();

                $(".box-fileupload__lable").text("Upload");

                var file = e.originalEvent.dataTransfer.files;

                DragAndDropUpload(file)
            });
        })
        $(settings.btnUpload).click(function () {

        })
        $(document).on('click', settings.deleteImageBtn, function () {
            let dataId = $(this).attr("data-id")
            removeElement(dataId);
            $(this).parent().remove()
        });
        return this;

    }
}(jQuery))
class TableCsv {
    /**
     * @param {HTMLTableElement} root The table element which will display the CSV data.
     */
    constructor(root) {
      this.root = root;
    }
  
    /**
     * Clears existing data in the table and replaces it with new data.
     *
     * @param {string[][]} data A 2D array of data to be used as the table body
     * @param {string[]} headerColumns List of headings to be used
     */
    update(data, headerColumns = []) {
      this.clear();
      this.setHeader(headerColumns);
      this.setBody(data);
    }
  
    /**
     * Clears all contents of the table (incl. the header).
     */
    clear() {
      this.root.innerHTML = "";
    }
  
    /**
     * Sets the table header.
     *
     * @param {string[]} headerColumns List of headings to be used
     */
    setHeader(headerColumns) {
      this.root.insertAdjacentHTML(
        "afterbegin",
        `
              <thead>
                  <tr>
                      ${headerColumns.map((text) => `<th>${text}</th>`).join("")}
                  </tr>
              </thead>
          `
      );
    }
  
    /**
     * Sets the table body.
     *
     * @param {string[][]} data A 2D array of data to be used as the table body
     */
    setBody(data) {
      const rowsHtml = data.map((row) => {
        return `
                  <tr>
                      ${row.map((text) => `<td>${text}</td>`).join("")}
                  </tr>
              `;
      });
  
      this.root.insertAdjacentHTML(
        "beforeend",
        `
              <tbody>
                  ${rowsHtml.join("")}
              </tbody>
          `
      );
    }
  }
  
  const tableRoot = document.querySelector("#csvRoot");
  const csvFileInput = document.querySelector("#csvFileInput");
  const tableCsv = new TableCsv(tableRoot);
  
  csvFileInput.addEventListener("change", (e) => {
    Papa.parse(csvFileInput.files[0], {
      delimiter: ",",
      skipEmptyLines: true,
      complete: (results) => {
        tableCsv.update(results.data.slice(1), results.data[0]);
      }
    });
  });
  