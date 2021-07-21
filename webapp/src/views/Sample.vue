<template>
  <div>
    <div class="bg">
      <b-container class="bv-example-row">
        <b-row class="mb-3"></b-row>
        <b-row class="mb-3" style="width: 25%; margin: 0 auto;">
          <b-col>
            <b-button size="lg" variant="outline-dark" @click="prevQuestion()" :disabled="questionId_idx == 0">Previous</b-button>
          </b-col>
          <b-col>
            <b-button size="lg" variant="outline-dark" @click="nextQuestion()" :disabled="questionId_idx == questionIds.length - 1">Next</b-button>
          </b-col>
        </b-row>
        <hr>
        <b-row>
          <b-col class="mb-12">
          </b-col>
        </b-row>
        <b-row class="mb-3">
          <b-col>
            <!-- <b-img thumbnail :src="require(`${image_path}`)" fluid alt="Responsive image"></b-img> -->
            <b-img thumbnail v-bind:src="image_path" fluid alt="Responsive image"></b-img>
          </b-col>
          <b-col>
            <b-card bg-variant="light" text-variant="dark" title="Statistics">
              <b-card-text>
                <b-row class="mb-3">
                  <b-col>
                    Question
                  </b-col>
                  <b-col cols="8">
                    {{ sample.question }}
                  </b-col>
                </b-row>
                <b-row>
                  <b-col>
                     Answer Id
                  </b-col>
                  <b-col>
                    Answer
                  </b-col>
                  <b-col>
                    Probability
                  </b-col>
                </b-row>
                <hr>
                <b-row v-for="(pred, index, i) in sample.predictions" :key="i">
                  <b-col>
                    {{ i+1 }}
                  </b-col>
                  <b-col :style="textColor(pred.answer, sample.ground_truth)">
                      {{ pred.answer }}
                  </b-col>
                  <b-col>
                      {{ pred.probability*100 | round }} %
                  </b-col>
                </b-row>
                <b-row class="mb-3"></b-row>
                <b-row>
                  <b-col>
                     Id
                  </b-col>
                  <b-col>
                    Ground Truth
                  </b-col>
                  <b-col>
                    Score
                  </b-col>
                </b-row>
                <hr>
                <b-row v-for="(ans, i) in sample.ground_truth" :key="i">
                  <b-col>
                    {{ i+1 }}
                  </b-col>
                  <b-col>
                    {{ ans.answer }}
                  </b-col>
                  <b-col>
                    {{ ans.score }}
                  </b-col>
                </b-row>
                <b-row class="mb-3"></b-row>
                <b-row class="mb-3">
                  <b-col>
                    Question Id
                  </b-col>
                  <b-col cols="8">
                    {{ sample.question_id }}
                  </b-col>
                </b-row>
              </b-card-text>
            </b-card>
          </b-col>
        </b-row>

        <b-row class="mb-3">
          <b-col>
            <b-card bg-variant="light" text-variant="dark" title="Image Bias Word Space">
              <b-row class="mb-3">
                  <b-col>
                    Score
                  </b-col>
                  <b-col cols="8">
                    {{ sample.image_bias_wordspace.score | round }} %
                  </b-col>
                </b-row>
                <b-row>
                  <b-col>
                    Answer Id
                  </b-col>
                  <b-col>
                    Answer
                  </b-col>
                  <b-col>
                    Frequency
                  </b-col>
                </b-row>
                <hr>
                <b-row v-for="(pred, i) in sample.image_bias_wordspace.predictions" :key="i">
                  <b-col>
                    {{ i+1 }}
                  </b-col>
                  <b-col :style="textColor(pred.answer, sample.ground_truth)">
                      {{ pred.answer }}
                  </b-col>
                  <b-col>
                      {{ pred.frequency*100 | round }} %
                  </b-col>
                </b-row>
            </b-card>
          </b-col>
        </b-row>

        <b-row class="mb-3">
          <b-col>
            <b-card bg-variant="light" text-variant="dark" title="Image Robustness Feature Space">
              <b-row class="mb-3">
                  <b-col>
                    Score
                  </b-col>
                  <b-col cols="8">
                    {{ sample.image_robustness_featurespace.score | round }} %
                  </b-col>
                </b-row>
                <b-row>
                  <b-col>
                    Answer Id
                  </b-col>
                  <b-col>
                    Answer
                  </b-col>
                  <b-col>
                    Frequency
                  </b-col>
                </b-row>
                <hr>
                <b-row v-for="(pred, i) in sample.image_robustness_featurespace.predictions" :key="i">
                  <b-col>
                    {{ i+1 }}
                  </b-col>
                  <b-col :style="textColor(pred.answer, sample.ground_truth)">
                      {{ pred.answer }}
                  </b-col>
                  <b-col>
                      {{ pred.frequency*100 | round }} %
                  </b-col>
                </b-row>
            </b-card>
          </b-col>
        </b-row>

        <b-row class="mb-3">
          <b-col>
            <b-card bg-variant="light" text-variant="dark" title="Image Robustness Image Space">
              <b-row class="mb-3">
                  <b-col>
                    Score
                  </b-col>
                  <b-col cols="8">
                    {{ sample.image_robustness_imagespace.score | round }} %
                  </b-col>
                </b-row>
                <b-row>
                  <b-col>
                    Answer Id
                  </b-col>
                  <b-col>
                    Answer
                  </b-col>
                  <b-col>
                    Frequency
                  </b-col>
                </b-row>
                <hr>
                <b-row v-for="(pred, i) in sample.image_robustness_imagespace.predictions" :key="i">
                  <b-col>
                    {{ i+1 }}
                  </b-col>
                  <b-col :style="textColor(pred.answer, sample.ground_truth)">
                      {{ pred.answer }}
                  </b-col>
                  <b-col>
                      {{ pred.frequency*100 | round }} %
                  </b-col>
                </b-row>
            </b-card>
          </b-col>
        </b-row>

        <b-row class="mb-3">
          <b-col>
            <b-card bg-variant="light" text-variant="dark" title="Question Bias Image Space">
              <b-row class="mb-3">
                  <b-col>
                    Score
                  </b-col>
                  <b-col cols="8">
                    {{ sample.question_bias_imagespace.score | round }} %
                  </b-col>
                </b-row>
                <b-row>
                  <b-col>
                    Answer Id
                  </b-col>
                  <b-col>
                    Answer
                  </b-col>
                  <b-col>
                    Frequency
                  </b-col>
                </b-row>
                <hr>
                <b-row v-for="(pred, i) in sample.question_bias_imagespace.predictions" :key="i">
                  <b-col>
                    {{ i+1 }}
                  </b-col>
                  <b-col :style="textColor(pred.answer, sample.ground_truth)">
                      {{ pred.answer }}
                  </b-col>
                  <b-col>
                      {{ pred.frequency*100 | round }} %
                  </b-col>
                </b-row>
            </b-card>
          </b-col>
        </b-row>

        <b-row class="mb-3">
          <b-col>
            <b-card bg-variant="light" text-variant="dark" title="Question Robustness Feature Space">
              <b-row class="mb-3">
                  <b-col>
                    Score
                  </b-col>
                  <b-col cols="8">
                    {{ sample.question_robustness_featurespace.score | round }} %
                  </b-col>
                </b-row>
                <b-row>
                  <b-col>
                    Answer Id
                  </b-col>
                  <b-col>
                    Answer
                  </b-col>
                  <b-col>
                    Frequency
                  </b-col>
                </b-row>
                <hr>
                <b-row v-for="(pred, i) in sample.question_robustness_featurespace.predictions" :key="i">
                  <b-col>
                    {{ i+1 }}
                  </b-col>
                  <b-col :style="textColor(pred.answer, sample.ground_truth)">
                      {{ pred.answer }}
                  </b-col>
                  <b-col>
                      {{ pred.frequency*100 | round }} %
                  </b-col>
                </b-row>
            </b-card>
          </b-col>
        </b-row>

        <b-row class="mb-3">
          <b-col>
            <b-card bg-variant="light" text-variant="dark" title="Semantically Equivalent Adversarial Rules (SEARs)">
                <b-row>
                  <b-col>
                    SEAR
                  </b-col>
                  <b-col>
                    Answer
                  </b-col>
                  <b-col>
                    Applied
                  </b-col>
                  <b-col>
                    Flipped
                  </b-col>
                </b-row>
                <hr>
                <b-row v-for="(sear, _, i) in sample.sears" :key="i">
                  <b-col>
                    {{ i+1 }}
                  </b-col>
                  <b-col :style="textColor(sear.answer, sample.ground_truth)">
                      {{ sear.answer }}
                  </b-col>
                  <b-col>
                      {{ sear.applied | translateBinary}}
                  </b-col>
                  <b-col>
                      {{ sear.flipped | translateBinary}}
                  </b-col>
                </b-row>
            </b-card>
          </b-col>
        </b-row>

        <b-row class="mb-3">
          <b-col>
            <b-card bg-variant="light" text-variant="dark" title="Uncertainty">
              <b-row>
                  <b-col>
                    Id
                  </b-col>
                  <b-col>
                    Answer
                  </b-col>
                  <b-col>
                    Frequency
                  </b-col>
                  <b-col>
                    Uncertainty
                  </b-col>
                </b-row>
                <hr>
                <b-row v-for="(uc, i) in sample.uncertainty" :key="i">
                  <b-col>
                    {{ i+1 }}
                  </b-col>
                  <b-col :style="textColor(uc.answer, sample.ground_truth)">
                      {{ uc.answer }}
                  </b-col>
                  <b-col>
                      {{ uc.frequency*100 | round }} %
                  </b-col>
                  <b-col>
                      {{ uc.uncertainty | round }} %
                  </b-col>
                </b-row>
            </b-card>
          </b-col>
        </b-row>
      </b-container>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      sampledata: [],
      questionId: "",
      counter: 0,
      questionIds: [],
      sample: {},
      image_path: "",
    };
  },
  created() {
    let temp = this.$route.params;
    this.questionId = temp.questionId;
    this.questionIds = temp.questionIds;
    this.questionId_idx = this.questionIds.findIndex(id => id == this.questionId);
    this.model = temp.model;
    this.dataset = temp.dataset
    this.getData(this.questionId);
  },  
  mounted() {
    // this.drawRect();
  },
  filters: {
    round: function (value) {
      return Math.round(value*100)/100
    },
    translateBinary: function (value) {
      if (value === 0) return false
      if (value === 1) return true
      return value
    }
  },
  methods: {
      nextQuestion() {
        this.questionId_idx += 1;
        this.getData(this.questionIds[this.questionId_idx]);
      },
      prevQuestion() {
        this.questionId_idx -= 1;
        this.getData(this.questionIds[this.questionId_idx]);
      },
      setImagePath(image_id, dataset) {
        var image = new Image();
        var url_image = '../images/' + image_id + '.jpg';
        image.src = url_image;
        if (image.width == 0) {
          this.image_path = require(`../images/2.jpg`);
        } else {
          this.image_path = require(`${dataset}' + '/images/' + ${image_id} + '.jpg`);
        }
      },
      getData(question_id) {
      let that = this;
      fetch('http://localhost:44123/sample', {method: 'POST', 
                                                  headers: {
                                                'Content-Type': 'application/json'
                                              },
                                              body: JSON.stringify({
                                                'model': this.model,
                                                'dataset': this.dataset,
                                                'questionId': question_id,
                                              }) 
      }).then(res => res.json()).then(function(data) {
        that.sample = data;
        that.setImagePath(data.imageId[0]);
      });
      
  },
  textColor: function (pred, ground_truth) {
      let bool = false;
      for (let i = 0; i < ground_truth.length; i++) { 
        if (pred == ground_truth[i].answer) {
          bool = true;
        }
      }
      if (bool) {
        return {
          color: 'green'
        }
      } else {
        return {
          color: 'red',
        }
      }
    },
    drawRect() {
      const data = this.filterData;
      var img = new Image();
      img.src = `data:image/png;base64,${data.sample.image}`;
      var c = document.getElementById("myCanvas");
      var ctx = c.getContext("2d");
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.drawImage(img, 0, 0, 300, 200);
      for (let i = 0; i < 3; i += 1) {
        let d = data.imageFeatures[i];
        ctx.beginPath();
        ctx.rect(d.x, d.y, d.width, d.height);
        ctx.font = "15px Arial";
        ctx.strokeText(d.label, d.x, d.y);
        ctx.stroke();
      }
    },
    changeData(key) {
      if (this.counter < this.questionIds.length) {
        if (key === "previous" && this.counter !== 0) {
          this.counter -= 1;
        } else {
          this.counter += 1;
        }
      } else {
        this.counter = 0;
      }
      this.questionid = this.questionIds[this.counter];
      this.getChart();
      this.drawRect();
    },
  },
};
</script>
